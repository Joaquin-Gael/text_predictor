from typing import Optional

import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import pandas as pd

import polars as pl

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from pathlib import Path

from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper

import itertools

from utils import get_data, decode_text, encode_text, only_spanish_letters, console, gen_graphic

console.print(th.cuda.is_available(), style="bold green")
console.print(th.cuda.get_device_name(0) if th.cuda.is_available() else "No CUDA available", style="bold green")
console.print(f"allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GiB")
console.print(f"reserved: {th.cuda.memory_reserved() / 1024**3:.2f} GiB")
console.print(th.cuda.memory_summary(device="cuda", abbreviated=True))

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class WikiDataset(Dataset):
    def __init__(self, tokens: int = 20):
        super().__init__()
        self.data: Optional[pl.DataFrame] = get_data(tokens=tokens)
        self.tokens = tokens

    def __len__(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return 0
        return len(self.data)

    def shape(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return (0, 0)
        return self.data.shape

    def __getitem__(self, idx):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            raise RuntimeError("Dataset is empty or not loaded.")

        tokens = []

        p0 = self.data.select("pred0")[idx].item()
        t0 = self.data.select("token0")[idx].item()
        tokens.append(t0)

        for i in range(1, self.tokens):
            t = self.data.select(f"token{i}")[idx].item()
            tokens.append(t)

        t1 = self.data.select("token1")[idx].item()
        tokens.append(t1)

        x = th.tensor(tokens, dtype=th.long)
        y = th.tensor([p0], dtype=th.long)

        return x, y

    @property
    def words(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return []
        
        list_words = list(itertools.chain(*self.data.select("token0").to_numpy().tolist()))
        for i in range(1, self.tokens):
            list_words.extend(list(itertools.chain(*self.data.select(f"token{i}").to_numpy().tolist())))

        console.print(f"Typo de list_words: {type(list_words)}", style="bold green")
        console.print(f"Total de palabras únicas: {len(set(list_words))}", style="bold green")
        
        return list_words

    @property
    def labels(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return []
        plane_words = list(itertools.chain(*self.data.select("pred0").to_numpy().tolist()))
        return plane_words

    @property
    def get_words(self):
        return decode_text(self.words)


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_token=0):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token)
        self.vocab_size = vocab_size

    def forward(self, input_words):
        seq_len = input_words.size(1)

        if seq_len > self.vocab_size:
            raise ValueError("Sequence length > max_len (pos emb)")

        pos_id = th.arange(seq_len, dtype=th.long).to(device).unsqueeze(0)
        pos_emb = self.embedding(pos_id)
        return pos_emb + input_words

class ComputeBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.3):
        super(ComputeBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.multy_head_attention = nn.MultiheadAttention(emb_size, num_heads=8, batch_first=True, dropout=0.3)
        self.FFN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, emb, attn_mask):
        norm = self.layer_norm(emb)
        attention_out, _ = self.multy_head_attention(norm, norm, norm, need_weights=False, attn_mask=attn_mask)
        residual = attention_out + emb
        norm_out = self.layer_norm(residual)
        ffn_out = self.FFN(norm_out)
        return ffn_out + residual

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, emb_size)
        self.model_blocks = nn.ModuleList([ComputeBlock(emb_size, hidden_size) for _ in range(num_steps)])
        self.norm = nn.LayerNorm(hidden_size)
        #self.output_layer = nn.Linear(hidden_size, output_size, bias=False)

        #self.output_layer.weight = self.embedding.weight

    def forward(self, input_words):
        emb = self.embedding(input_words)
        enc = self.positional_encoding(emb)

        _mask = th.triu(
            th.ones(
                (input_words.size(1), input_words.size(1)),
                dtype=th.bool,
                device=device
            )
        )

        for block in self.model_blocks:
            out = block(enc, attn_mask=_mask)

        #norm = self.norm(enc)
        #out = self.output_layer(norm)
        return out

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps):
        super(GPT2Model, self).__init__()
        self.transformers = nn.ModuleList([TransformerModel(vocab_size, emb_size, hidden_size, output_size, num_steps) for _ in range(num_steps)])
        self.output_layer = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input_words):
        for transformer in self.transformers:
            out = transformer(input_words)
        out = self.output_layer(out)
        return out

class NanoModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps, head_num, mem_len=512):
        super(NanoModel, self).__init__()
        
        self.mem_len = mem_len
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = PositionalEncoding(vocab_size, emb_size)
        
        # Relative positional encoding
        self.u = nn.Parameter(th.Tensor(head_num, emb_size // head_num))
        self.v = nn.Parameter(th.Tensor(head_num, emb_size // head_num))
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)
        
        # Transformer-XL blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerXLBlock(
                emb_size,
                head_num,
                hidden_size,
                dropout=0.1
            ) for _ in range(num_steps)
        ])
        
        self.output_layer = nn.Linear(emb_size, output_size, bias=False)
        self.output_layer.weight = self.embedding.weight
        
        # Memory states
        self.mems = []
        
    def _update_mems(self, hidden_states, mems):
        # Update memory with current hidden states
        if len(mems) == 0:
            mems = [th.empty(0, dtype=hidden_states.dtype, device=device)] * len(self.transformer_blocks)
        assert len(hidden_states) == len(mems)
        
        with th.no_grad():
            new_mems = []
            for i in range(len(hidden_states)):
                cat = th.cat([mems[i], hidden_states[i]], dim=1)
                new_mems.append(cat[:, -self.mem_len:].detach())
        return new_mems

    def forward(self, input_words, mems=None):
        if mems is None:
            mems = [th.empty(0, dtype=th.float, device=device)] * len(self.transformer_blocks)
            
        # Word embeddings + positional encoding
        word_emb = self.embedding(input_words)
        pos_emb = self.pos_emb(word_emb)
        
        # Create causal attention mask
        seq_len = input_words.size(1)
        attn_mask = th.triu(
            th.ones((seq_len, seq_len), dtype=th.bool, device=device),
            diagonal=1
        )
        
        hidden_states = []
        current_hidden = pos_emb
        
        # Process through transformer blocks with memory
        for transformer_block, m in zip(self.transformer_blocks, mems):
            # Extend attention mask for memory tokens
            if m.numel() > 0:
                mem_len = m.size(1)
                mem_mask = th.zeros((seq_len, mem_len), dtype=th.bool, device=device)
                attn_mask_extended = th.cat([mem_mask, attn_mask], dim=1)
            else:
                attn_mask_extended = attn_mask
                
            current_hidden = transformer_block(
                current_hidden,
                m,
                self.u,
                self.v,
                attn_mask_extended
            )
            hidden_states.append(current_hidden)
            
        # Update memory states
        new_mems = self._update_mems(hidden_states, mems)
        
        # Output projection
        output = self.output_layer(current_hidden)
        
        return output, new_mems

class TransformerXLBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.ff = nn.Sequential(
            nn.Linear(emb_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mem, u, v, mask=None, max_mem_len=512):
        norm = self.norm(x)
        
        B, L, E = norm.shape

        
        if th.isnan(norm).any().item():
            print("NaN values in norm")


        if mem.numel() > 0:
            ctx = th.cat([mem, norm], dim=1)[:, -max_mem_len:, :]
            
        else:
            ctx = norm


        attended, _ = self.attention(
            norm,
            ctx,
            ctx,
            attn_mask=mask[:, -max_mem_len:]
        )
        x = x + attended
        
        # Feed forward
        x = x + self.ff(self.norm2(x))
        return x
    

dataset = WikiDataset(tokens=40)

train_dataset, test_dataset = random_split(dataset, [int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])

train_dataset: th.Tensor = train_dataset
test_dataset: th.Tensor = test_dataset

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = max(dataset.words)+1
emb_size = 600
output_size = vocab_size
hidden_size = 600
num_steps = 10
num_heads = 10

num_epochs = 1

model = NanoModel(vocab_size, emb_size, hidden_size, output_size, num_steps, num_heads)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0001)


if __name__ == "__main__":
    print(f"Size Data Set: {dataset.shape()}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Embedding Size: {emb_size}")
    print(f"Output Size: {output_size}")
    print(f"Input Size: {emb_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Word Size: {test_dataset[1][0].tolist()}")
    print(f"Word Decode: {decode_text(test_dataset[1][0].tolist())}")

    with th.inference_mode():
        sequence_input = "Argentina es una tierra de"
        encode = encode_text(sequence_input)
        input_tensor = th.tensor([encode], dtype=th.long)
        
        out_put_ids: list[int] = []
        
        input_tensor = input_tensor.to(device)

        out, mems = model(input_tensor)

        predicted_indices = out.argmax(dim=-1).tolist()[0]
        out_put_ids.extend(predicted_indices)
        sequence_output = sequence_input + decode_text(predicted_indices)
        
        for i in range(10):
            input_tensor_i = th.tensor([encode_text(sequence_output)], dtype=th.long).to(device)
            
            
            out, mems = model(input_tensor_i, mems)

            
            last_logits = out[:, -2, :]
            
            processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(penalty=1.2)
            ])
            
            warped_logits = processor(input_ids=input_tensor, scores=last_logits)
            
            warper = TopPLogitsWarper(top_p=0.9)
            warped_logits = warper(input_tensor, warped_logits)
            
            probs = th.softmax(warped_logits, dim=-1)
            next_id = th.multinomial(probs, num_samples=1)
            
            out_put_ids.append(next_id.item())
            
            input_tensor = th.cat([input_tensor, next_id], dim=1)
            
        console.print("Out Put Ids:", out_put_ids)

        decode_data = decode_text(out_put_ids)
        
        console.print("Decode Ids:", decode_data)

        console.print("Original Sequence:", decode_data)
            
        console.print("Final Sequence only spanish:", only_spanish_letters(decode_data))


    for epoch in range(num_epochs):
        model.train()
        total_loss_train = []

        with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("[green]{task.fields[info]}"),
                TextColumn("[red]Loss: {task.fields[loss]}"),
                TextColumn("[red]Loss mean: {task.fields[loss_mean]}"),
                TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(
                description=f"Epoch {epoch} - Train",
                total=len(train_dataloader),
                info="Iniciando",
                loss="None",
                loss_mean="None",
            )

            for idx, batch in enumerate(train_dataloader):
                progress.update(task, info=f"Lote {idx}")

                input_words = batch[0]
                target = batch[1]
                input_words = input_words.to(device)
                target = target.to(device)
                mems = None
                try:
                    if not mems is None:
                        out, mems = model(input_words, mems)
                    else:
                        out, mems = model(input_words)
                        
                    if th.isnan(out).any().item():
                        console.print("NaN en out!")

                except Exception as e:
                    console.print_exception(show_locals=True)
                    console.print("Error:", e)
                    console.print("Word:", input_words)
                    continue
                

                loss = criterion(out[:, -2, :], target.squeeze(-1).long())
                total_loss_train.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress.update(
                    task,
                    loss=f"{loss.item():.4f}",
                    completed=float(idx),
                    loss_mean=f"{sum(total_loss_train) / len(total_loss_train)}"
                )

        model.eval()
        total_loss_test = []

        with Progress(
                SpinnerColumn(),
                TextColumn("[bold red]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("[green]{task.fields[info]}"),
                TextColumn("[blue]Loss: {task.fields[loss]}"),
                TextColumn("[blue]Loss mean: {task.fields[loss_mean]}"),
                TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(
                description=f"Epoch {epoch} - Test",
                total=len(test_dataloader),
                info="Iniciando",
                loss="None",
                loss_mean="None",
            )

            for idx, batch in enumerate(test_dataloader):
                progress.update(task, info=f"Lote {idx}")

                input_words = batch[0]
                target = batch[1]
                input_words = input_words.to(device)
                target = target.to(device)
                memsm = None
                
                try:
                    if not mems is None:
                        out, mems = model(input_words, mems)
                    else:
                        out, mems = model(input_words)
                except Exception as e:
                    print("Error:", e)
                    print("Word:", input_words)
                    continue

                loss = criterion(out[:, -2, :], target.squeeze(-1).long())
                total_loss_test.append(loss.item())
                progress.update(
                    task,
                    loss=f"{loss.item():.4f}",
                    completed=float(idx),
                    loss_mean=f"{sum(total_loss_test) / len(total_loss_test)}"
                )


        print(f"Epoch {epoch}, Error: {sum(total_loss_test) / len(total_loss_test)}")



        diff = len(total_loss_test) - len(total_loss_train)

        if not len(total_loss_test) > len(total_loss_train):
            diff = len(total_loss_train) - len(total_loss_test)

        if diff > 0:
            total_loss_test_extended = total_loss_test + [np.nan] * diff
        else:
            total_loss_test_extended = total_loss_test

        df_loss = pd.DataFrame({
            "Epoch":range(1, len(total_loss_train)+1),
            "Loss Train": total_loss_train,
            "Loss Test":total_loss_test_extended
        })

        fig = gen_graphic(
            df_loss,
            len(total_loss_train),
            "Loss",
        )

        fig.show()



    with th.inference_mode():
        sequence_input = "Argentina es"
        encode = encode_text(sequence_input)
        input_tensor = th.tensor([encode], dtype=th.long)
        
        out_put_ids: list[int] = []
        
        input_tensor = input_tensor.to(device)

        out, mems = model(input_tensor)

        predicted_indices = out.argmax(dim=-1).tolist()[0]
        out_put_ids.extend(predicted_indices)
        sequence_output = sequence_input + decode_text(predicted_indices)
        
        for i in range(10):
            input_tensor_i = th.tensor([encode_text(sequence_output)], dtype=th.long).to(device)
            
            out, mems = model(input_tensor_i, mems)
            
            last_logits = out[:, -2, :]
            
            processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(penalty=1.2)
            ])
            
            warped_logits = processor(input_ids=input_tensor, scores=last_logits)
            
            warper = TopPLogitsWarper(top_p=0.9)
            warped_logits = warper(input_tensor, warped_logits)
            
            probs = th.softmax(warped_logits, dim=-1)
            next_id = th.multinomial(probs, num_samples=1)
            
            out_put_ids.append(next_id.item())
            
            input_tensor = th.cat([input_tensor, next_id], dim=1)
            
        console.print("Out Put Ids:", out_put_ids)

        decode_data = decode_text(out_put_ids)
        
        console.print("Decode Ids:", decode_data)

        console.print("Original Sequence:", decode_data)
            
        console.print("Final Sequence only spanish:", only_spanish_letters(decode_data))
        
    if df_loss["Loss Train"].isnull().all():
        console.print("No hay datos de pérdida de entrenamiento para mostrar.", style="bold red")
        
    elif df_loss["Loss Test"].mean() < 5:
        console.print("El modelo ha convergido con éxito.", style="bold green")
        th.save(model.state_dict(), "model.pth")
        console.print("Modelo guardado como 'model.pth'.", style="bold green")