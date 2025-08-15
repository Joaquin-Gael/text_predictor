from typing import Optional
from utils import console, gen_graphic
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

from utils import get_data, decode_text, encode_text, only_spanish_letters

console.print(th.cuda.is_available(), style="bold green")
console.print(th.cuda.get_device_name(0) if th.cuda.is_available() else "No CUDA available", style="bold green")

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

dataset = WikiDataset()

train_dataset, test_dataset = random_split(dataset, [int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])

train_dataset: th.Tensor = train_dataset
test_dataset: th.Tensor = test_dataset

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = max(dataset.words)+1
emb_size = 256
output_size = vocab_size
hidden_size = 256
num_steps = 5

num_epochs = 1

model = GPT2Model(vocab_size, emb_size, hidden_size, output_size, num_steps)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)


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
        sequence_input = "Argentina es"
        encode = encode_text(sequence_input)
        input_tensor = th.tensor([encode], dtype=th.long)
        
        input_tensor = input_tensor.to(device)

        out = model(input_tensor)

        predicted_indices = out.argmax(dim=-1).tolist()[0]
        console.print("Predicted Indices:", predicted_indices, style="bold green")
        console.print("out.shape:", out.shape)
        console.print("input_tensor.shape:", input_tensor.shape)
        console.print("out.min(), out.max():", out.min().item(), out.max().item())
        sequence_output = sequence_input + decode_text(predicted_indices)

        print("Oración final:", sequence_output)


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
                try:
                    out = model(input_words)
                except Exception as e:
                    print("Error:", e)
                    print("Word:", input_words)
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

                try:
                    out = model(input_words)
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

        out = model(input_tensor)

        predicted_indices = out.argmax(dim=-1).tolist()[0]
        out_put_ids.extend(predicted_indices)
        sequence_output = sequence_input + decode_text(predicted_indices)
        
        for i in range(10):
            out = model(th.tensor([encode_text(sequence_output)], dtype=th.long).to(device))
            
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

        console.print("Original Sequence:", sequence_input + decode_data)
            
        console.print("Final Sequence only spanish:", only_spanish_letters(sequence_input + decode_data))
        
    if df_loss["Loss Train"].isnull().all():
        console.print("No hay datos de pérdida de entrenamiento para mostrar.", style="bold red")
        
    elif df_loss["Loss Test"].mean() < 5:
        console.print("El modelo ha convergido con éxito.", style="bold green")
        th.save(model.state_dict(), "model.pth")
        console.print("Modelo guardado como 'model.pth'.", style="bold green")