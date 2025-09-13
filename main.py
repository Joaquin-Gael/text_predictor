from turtle import forward
import torch as th
from torch import hsmm, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import math as mt

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

from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    NoRepeatNGramLogitsProcessor,
)

import itertools

from utils import decode_text, encode_text, only_spanish_letters, console, gen_graphic

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def get_device(train: bool = False):
    try:
        if train:
            return device
        return th.device("cpu")
    except Exception:
        return device


class WikiDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.data = pl.read_csv(f"./data/{csv_path}")
        self.tokens = len(self.data.columns)-1

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

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-mt.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, input_words: th.Tensor) -> th.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = input_words + self.pe[:input_words.size(0)]
        return self.dropout(x)

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

        # Build mask on the same device as the inputs to avoid device mismatches
        _mask = th.triu(
            th.ones(
                (input_words.size(1), input_words.size(1)),
                dtype=th.bool,
                device=input_words.device
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

class LineAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1) -> None:
        super(LineAttention, self).__init__()
        

class TransformerXLBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.projection = nn.Linear(emb_size, emb_size)
                
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
        
    def forward(self, x, mem, mask=None, max_mem_len=512):
        norm = self.norm(x)
        
        B, L, E = norm.shape

        # Alinear memoria al batch/emb actual
        if mem.numel() > 0:
            if mem.dim() != 3 or mem.size(0) != B or mem.size(2) != E:
                mem = norm[:, 0:0, :]  # (B, 0, E)
        else:
            # asegurar tensor 3D vacío consistente
            mem = norm[:, 0:0, :]

        if th.isnan(norm).any().item():
            print("NaN values in norm")

        if mem.numel() > 0:
            MEM_B, MEM_L, MEM_E = mem.shape
            if MEM_L == 0:
                ctx = self.projection(th.cat([mem, norm], dim=1)[:, -L:, :])
            else:
                ctx = self.projection(th.cat([mem, norm], dim=1)[:, -MEM_L:, :])
        else:
            ctx = self.projection(norm)
            MEM_L = L
        
        # Máscara debe coincidir con (L, MEM_L) o (L, L)
        if mask is not None:
            total_k = ctx.size(1)
            attn_mask = mask[:, :total_k]
        else:
            attn_mask = None

        attended, _ = self.attention(
            norm,
            ctx,
            ctx,
            attn_mask=attn_mask
        )
        x = x + attended
        
        x = x + self.ff(self.norm2(x))
        
        return x

class LineTransformerXLHRMBlock(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout, batch_first=True) # Por modificar
        
        self.norm = nn.LayerNorm(emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        self.projection = nn.Linear(emb_size, emb_size)
                
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
        
    def forward(self, x, mem_h, mem_l, mask=None, max_mem_len=512):
        norm = self.norm(x)
        
        B, L, E = norm.shape

        # Alinear memoria al batch/emb actual
        if mem_l.numel() > 0:
            if mem_l.dim() != 3 or mem_l.size(0) != B or mem_l.size(2) != E:
                mem_l = norm[:, 0:0, :]  # (B, 0, E)
        else:
            # asegurar tensor 3D vacío consistente
            mem_l = norm[:, 0:0, :]

        if mem_h.numel() > 0:
            if mem_h.dim() != 3 or mem_h.size(0) != B or mem_h.size(2) != E:
                mem_h = norm[:, 0:0, :]  # (B, 0, E)
        else:
            # asegurar tensor 3D vacío consistente
            mem_h = norm[:, 0:0, :]

        if th.isnan(norm).any().item():
            print("NaN values in norm")

        if mem_l.numel() > 0:
            MEM_B, MEM_L, MEM_E = mem_l.shape
            if MEM_L == 0:
                ctx = self.projection(th.cat([mem_l, norm], dim=1)[:, -L:, :])
            else:
                ctx = self.projection(th.cat([mem_l, norm], dim=1)[:, -MEM_L:, :])
        else:
            ctx = self.projection(norm)
            MEM_L = L

        if mem_h.numel() > 0:
            MEM_B, MEM_L, MEM_E = mem_h.shape
            if MEM_L == 0:
                ctx = self.projection(th.cat([mem_h, mem_l], dim=1)[:, -L:, :])
            else:
                ctx = self.projection(th.cat([mem_h, mem_l], dim=1)[:, -MEM_L:, :])
        else:
            ctx = self.projection(mem_l)
            MEM_L = L

        # Máscara debe coincidir con (L, MEM_L) o (L, L)
        if mask is not None:
            total_k = ctx.size(1)
            attn_mask = mask[:, :total_k]
        else:
            attn_mask = None

        attended, _ = self.attention(
            norm,
            ctx,
            ctx,
            attn_mask=attn_mask
        )
        x = x + attended
        
        x = x + self.ff(self.norm2(x))
        
        return x

class NanoModelHRM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps, head_num, dropout, mem_len=512) -> None:
        super(NanoModelHRM, self).__init__()

        self.mem_len = mem_len
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = PositionalEncoding(emb_size, dropout)

        self.H_T = TransformerXLBlock(emb_size, hidden_size, head_num, dropout)
        self.L_T = LineTransformerXLHRMBlock(emb_size, hidden_size, head_num, dropout)

        self.output_layer = nn.Linear(emb_size, output_size, bias=False)
        self.output_layer.weight = self.embedding.weight

        self.mems_h_l = [None, None]

    def forward(self, input_words: th.Tensor, mems: list[th.Tensor | None, th.Tensor | None] | None =None, N: int=2, T: int=2):
        if mems is None:
            mems = self.mems_h_l
        if mems[0] is None and mems[1] is None:
            B = input_words.size(0)
            dev = input_words.device
            dt = th.float
            mems[0] = th.zeros((B, 0, self.emb_size), dtype=dt, device=dev)
            mems[1] = th.zeros((B, 0, self.emb_size), dtype=dt, device=dev)

        word_emb = self.embedding(input_words)
        pos_emb = self.pos_emb(word_emb)

        with th.no_grad():
            for i in range(N * T - 1):
                mems[1] = self.L_T(pos_emb, mems[1], mems[0])
                if (i + 1) % T == 0:
                    mems[0] = self.H_T(mems[1], mems[0])

        mems[1] = self.L_T(pos_emb, mems[1], mems[0])
        mems[0] = self.H_T(mems[1], mems[0])

        out = self.output_layer(mems[0])

        return out, mems

class NanoModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps, head_num, dropout, mem_len=512):
        super(NanoModel, self).__init__()
        
        self.mem_len = mem_len
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = PositionalEncoding(emb_size, dropout)
        
        # Transformer-XL blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerXLBlock(
                emb_size,
                hidden_size,
                head_num,
                dropout=dropout
            ) for _ in range(num_steps)
        ])
        
        self.output_layer = nn.Linear(emb_size, output_size, bias=False)
        self.output_layer.weight = self.embedding.weight
        
        # Memory states
        self.mems = []
        
    def _update_mems(self, hidden_states, mems):
        # Update memory with current hidden states
        # mems y hidden_states deben coincidir en batch y emb
        with th.no_grad():
            new_mems = []
            for i in range(len(hidden_states)):
                hs = hidden_states[i]
                if i < len(mems):
                    m = mems[i]
                else:
                    m = hs[:, 0:0, :]  # (B, 0, E)

                # Normalizar shape de memoria
                if m.dim() != 3 or m.size(0) != hs.size(0) or m.size(2) != hs.size(2):
                    m = hs[:, 0:0, :]

                cat = th.cat([m, hs], dim=1)
                new_mems.append(cat[:, -self.mem_len:].detach())
        return new_mems

    def forward(self, input_words, mems=None):
        if mems is None:
            # inicializar mems vacías con shape (B, 0, E)
            B = input_words.size(0)
            dev = input_words.device
            dt = th.float
            mems = [th.zeros((B, 0, self.emb_size), dtype=dt, device=dev)] * len(self.transformer_blocks)
            
        # Word embeddings + positional encoding
        word_emb = self.embedding(input_words)
        pos_emb = self.pos_emb(word_emb)
        
        # Create causal attention mask
        seq_len = input_words.size(1)
        attn_mask = th.triu(
            th.ones((seq_len, seq_len), dtype=th.bool, device=input_words.device),
            diagonal=1
        )
        
        hidden_states = []
        current_hidden = pos_emb
        
        # Process through transformer blocks with memory
        for transformer_block, m in zip(self.transformer_blocks, mems):
            # Extend attention mask for memory tokens
            if m.numel() > 0:
                mem_len = m.size(1)
                mem_mask = th.zeros((seq_len, mem_len), dtype=th.bool, device=input_words.device)
                attn_mask_extended = th.cat([mem_mask, attn_mask], dim=1)
            else:
                attn_mask_extended = attn_mask
                
            current_hidden = transformer_block(
                current_hidden,
                m,
                attn_mask_extended
            )
            hidden_states.append(current_hidden)
            
        # Update memory states
        new_mems = self._update_mems(hidden_states, mems)
        
        # Output projection
        output = self.output_layer(current_hidden)
        
        return output, new_mems
    
    def generate(self, input_words, mems=None, max_tokens: int = 10):
        """Generar secuencia de tokens a partir de entrada."""
        loggits, mems = self(input_words, mems)
        out_ids = []
        for _ in range(max_tokens-1):
            console.print(f"Loggits shape: {loggits.shape}", style="bold green")
            loggits = loggits[:, -1, :]
            processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(penalty=1.3),
                NoRepeatNGramLogitsProcessor(ngram_size=1),
            ])
            warper = TopPLogitsWarper(top_p=0.9)
            loggits = processor(input_ids=input_words, scores=loggits)
            loggits = warper(input_ids=input_words, scores=loggits)
            probs = th.softmax(loggits, dim=-1)
            next_token = th.multinomial(probs, num_samples=1)
            out_ids.append(next_token.item())
            input_words = th.cat([input_words, next_token], dim=1)
            loggits, mems = self(input_words, mems)
        return out_ids

def text_test(model: NanoModel):
    with th.inference_mode():
        sequence_input = "Argentina es una tierra de"
        encode = encode_text(sequence_input)
        input_tensor = th.tensor([encode], dtype=th.long).to(get_device(train=True))
        
        out_put_ids: list[int] = []
        sequence_output = sequence_input

        # primer forward para inicializar mems
        out, mems = model(input_tensor)

        for _ in range(10):
            # usar el último logit correspondiente al último token del input actual
            last_logits = out[:, -1, :]
            processor = LogitsProcessorList([
                RepetitionPenaltyLogitsProcessor(penalty=1.3),
                NoRepeatNGramLogitsProcessor(ngram_size=1),
            ])

            warped_logits = processor(input_ids=input_tensor, scores=last_logits)
            warper = TopPLogitsWarper(top_p=0.9)
            warped_logits = warper(input_tensor, warped_logits)
            probs = th.softmax(warped_logits, dim=-1)
            next_id = th.multinomial(probs, num_samples=1)  # [B, 1]

            out_put_ids.append(next_id.item())

            # actualizar secuencia y entrada
            input_tensor = th.cat([input_tensor, next_id.to(input_tensor.device)], dim=1)
            sequence_output += decode_text([next_id.item()])

            # siguiente paso: sólo el nuevo token y memoria
            out, mems = model(next_id.to(get_device(train=True)), mems)

        console.print("Out Put Ids:", out_put_ids)

        decode_data = decode_text(out_put_ids)
        
        console.print("Decode Ids:", decode_data)

        console.print("Original Sequence:", sequence_output)
            
        console.print("Final Sequence only spanish:", only_spanish_letters(sequence_output))


def train_model(epochs:int, batch_size:int, learning_rate:float, csv_path:str, model_path:str, hidden_size:int, emb_size:int, dropout:float):
    
    console.print(th.cuda.get_device_name(0) if th.cuda.is_available() else "No CUDA available", style="bold green")
    console.print(f"allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GiB") if th.cuda.is_available() else None
    console.print(f"reserved: {th.cuda.memory_reserved() / 1024**3:.2f} GiB") if th.cuda.is_available() else None
    console.print(th.cuda.memory_summary(device="cuda", abbreviated=True) if th.cuda.is_available() else "Not Cuda")
    
    dataset = WikiDataset(csv_path)
    
    vocab_size = max(dataset.words)+1
    emb_size = 600
    output_size = vocab_size
    hidden_size = 600
    num_steps = 10
    num_heads = 10
    
    if model_path:
        model: NanoModelHRM = th.load(model_path)
    else:
        model: NanoModelHRM = NanoModelHRM(vocab_size, emb_size, hidden_size, output_size, num_steps, num_heads, dropout)

    train_dataset, test_dataset = random_split(dataset, [int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])

    train_dataset: th.Tensor = train_dataset
    test_dataset: th.Tensor = test_dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = epochs
    console.print(f"Model: {model}\n Device: {get_device(train=True)}", style="bold green")
    model = model.to(get_device(train=True))
    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        
    print(f"Size Data Set: {dataset.shape()}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Embedding Size: {emb_size}")
    print(f"Output Size: {output_size}")
    print(f"Input Size: {emb_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Word Size: {test_dataset[1][0].tolist()}")
    print(f"Word Decode: {decode_text(test_dataset[1][0].tolist())}")

    text_test(model)

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
                input_words = input_words.to(get_device(train=True))
                target = target.to(get_device(train=True))
                mems = None

                with th.amp.autocast("cuda"):
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
                input_words = input_words.to(get_device(train=True))
                target = target.to(get_device(train=True))
                mems = None
                
                with th.amp.autocast("cuda"):
                    try:
                        if not mems is None:
                            out, mems = model(input_words, mems)
                        else:
                            out, mems = model(input_words)
                    except Exception as e:
                        console.print("Error:", e)
                        console.print("Word:", input_words)
                        console.print_exception(show_locals=True)
                        continue

                    loss = criterion(out[:, -2, :], target.squeeze(-1).long())
                total_loss_test.append(loss.item())
                progress.update(
                    task,
                    loss=f"{loss.item():.4f}",
                    completed=float(idx),
                    loss_mean=f"{sum(total_loss_test) / len(total_loss_test)}"
                )


        console.print(f"Epoch {epoch}, Error: {sum(total_loss_test) / len(total_loss_test)}")


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

        model_name = f"model_{epochs}"

        if model_path:
            model_name = f"model_{int(model_path.strip("_.pth"))+epochs}"
        
        th.save(model.state_dict(), f"./models/{model_name}.pth")

        fig.show()
        

    text_test(model)
        
    if df_loss["Loss Train"].isnull().all():
        console.print("No hay datos de pérdida de entrenamiento para mostrar.", style="bold red")
        
    elif df_loss["Loss Test"].mean() < 5:
        console.print("El modelo ha convergido con éxito.", style="bold green")
        th.save(model.state_dict(), "model.pth")
        console.print("Modelo guardado como 'model.pth'.", style="bold green")