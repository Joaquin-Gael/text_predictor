from typing import Optional
from utils import console
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import pandas as pd

import polars as pl

import plotly.graph_objects as go

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
      
#writer = SummaryWriter(
 #  str(Path(__file__).parent.joinpath('logs').resolve()),
#)

class WikiDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data: Optional[pl.DataFrame] = get_data()

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
        # Extraer valores (polars permite indexar la columna como lista-like)
        t0 = int(self.data["token0"][idx])
        t1 = int(self.data["token1"][idx])

        p0 = self.data["pred0"][idx]

        # Definir pad/ignore id (usa -100 para CrossEntropy ignore_index)
        pad_id = getattr(self, "pad_token_id", -100)

        p0 = int(p0) if p0 is not None else pad_id

        # Tensores de entrada y target
        x = th.tensor([t0, t1], dtype=th.long)   # shape (2,)
        y = th.tensor([p0], dtype=th.long)   # shape (2,)

        return x, y

    @property
    def words(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return []
        
        list_words = list(itertools.chain(*self.data.select("token0").to_numpy().tolist()))
        list_words.extend(list(itertools.chain(*self.data.select("token1").to_numpy().tolist())))
        
        console.print(f"Typo de list_words: {type(list_words)}", style="bold green")
        console.print(f"Total de palabras únicas: {len(set(list_words))}", style="bold green")
        
        return list_words

    @property
    def labels(self):
        if self.data is None:
            console.print("Dataset is empty or not loaded.", style="bold red")
            return []
        plane_words = list(itertools.chain(*self.data.select("prediction").to_numpy().tolist()))
        return plane_words

    @property
    def get_words(self):
        return decode_text(self.words)


class WordRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps, pad_token=0):
        super(WordRNN, self).__init__()

        self.hidden_size = hidden_size # es la capa que se encarga de comunicar el contexto a las otras
        self.num_steps = num_steps # nose

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token) # un mapeo de las palabras pasandolas a vectores de 128 dimenciones
        self.input_fc = nn.Linear(emb_size, hidden_size)

        self.msg_fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.msg_fc2 = nn.Linear(hidden_size, hidden_size)

        self.update_fc = nn.Linear(2*hidden_size, hidden_size)

        self.out_fc = nn.Linear(hidden_size, output_size)

    def compute_message(self, h_i, h_j):
        if h_i.size(1) == 0 or h_j.size(1) == 0:
            print("Entrada vacía en compute_message.")
            return th.zeros(h_i.size(0), self.msg_fc1.out_features, device=h_i.device)

        msg_input = th.cat([h_i, h_j], dim=1)
        msg = F.relu(self.msg_fc1(msg_input))
        msg = F.relu(self.msg_fc2(msg))

        return msg

    def forward(self, input_words):
        batch_size, seq_len = input_words.size()

        #print("batch_size:", batch_size, "\n", " seq_len:", seq_len)

        if seq_len == 1:
            #print("Secuencia de longitud 1, ajustando el procesamiento.")
            emb = self.embedding(input_words)  # [1, 1, emb_size]
            h = F.relu(self.input_fc(emb))     # [1, 1, hidden_size]
            h_pool = th.mean(h, dim=1)
            out = self.out_fc(h_pool)
            return F.log_softmax(out, dim=1)

        emb = self.embedding(input_words)
        h = F.relu(self.input_fc(emb))

        for _ in range(self.num_steps):
            m = th.zeros_like(h)

            m[:,1:] += self.compute_message(h[:,1:], h[:,:-1])
            m[:, :-1] += self.compute_message(h[:,:-1], h[:, 1:])

            h = F.relu(self.update_fc(th.cat([h, F.relu(self.input_fc(emb))], dim=-1))+m)

        h_pool = th.mean(h, dim=1)
        out = self.out_fc(h_pool)

        out = F.log_softmax(out, dim=1)

        return out

class WordRRN_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, pad_token=0):
        super(WordRRN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=3, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=5, batch_first=True)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_words):
        emb = self.embedding(input_words)
        rnn_out, _ = self.rnn(emb)
        dropout = self.dropout(rnn_out)
        cov_out = self.conv(dropout.permute(0, 2, 1))  # Conv1d expects (batch_size, channels, seq_len)
        cov_out = cov_out.permute(0, 2, 1)  #
        lstm_out, _ = self.lstm(dropout)
        h_pool = th.mean(lstm_out, dim=1)
        x = self.fc(h_pool)
        out = self.softmax(x)
        return out

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, pad_token=0):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.multy_head_attention = nn.MultiheadAttention(emb_size, num_heads=8, batch_first=True, dropout=0.3)
        self.FFN = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(p=0.2)
        )
        self.out_put = nn.Sequential(
            nn.Linear(emb_size, output_size),
            nn.Softmax(dim=-1)
        )
        
    def block_model(self, emb):
        norm = self.layer_norm(emb)
        attention_out, _ = self.multy_head_attention(norm, norm, norm, need_weights=False)
        residual = attention_out + emb
        norm_out = self.layer_norm(residual)
        ffn_out = self.FFN(norm_out)
        return ffn_out + norm_out
    
    def forward(self, input_words):
        emb = self.embedding(input_words)
        block_out = self.block_model(emb)
        block_out = self.block_model(block_out)
        block_out = self.block_model(block_out)
        block_out = self.block_model(block_out)
        norm_ffn_out = self.layer_norm(block_out)
        out = self.out_put(norm_ffn_out)
        return out

dataset = WikiDataset()

train_dataset, test_dataset = random_split(dataset, [int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])

train_dataset: th.Tensor = train_dataset
test_dataset: th.Tensor = test_dataset

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = max(dataset.words)+1
emb_size = 128
output_size = vocab_size
hidden_size = 256
num_steps = 3

num_epochs = 300

model = TransformerModel(vocab_size, emb_size, hidden_size, output_size, pad_token=0)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.000003)

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
        console.print("out.shape:", out.shape)       # debe ser (batch, seq_len, vocab) si todo está bien
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
                #console.print("Input Words:", input_words, style="bold green")
                #console.print("Input Words Shape:", input_words.shape, style="bold green")
                target = batch[1]
                #console.print("Target:", target, style="bold green")
                #console.print("Target Shape:", target.shape, style="bold green")
                input_words = input_words.to(device)
                target = target.to(device)
                try:
                    out = model(input_words)
                except Exception as e:
                    print("Error:", e)
                    print("Word:", input_words)
                    continue

                loss = criterion(out[:, 0], target[:, 0])
                progress.update(task, loss=f"{loss.item():.4f}")
                #writer.add_scalar("train loss", loss.item(), idx)
                total_loss_train.append(loss.item())
                progress.update(task, loss_mean=f"{sum(total_loss_train) / len(total_loss_train)}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress.update(task ,completed=float(idx))

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

                loss = criterion(out[:, 0], target[:, 0])
                progress.update(task, loss=f"{loss.item():.4f}")
                #writer.add_scalar("test loss", loss.item(), idx)
                total_loss_test.append(loss.item())
                progress.update(task, loss_mean=f"{sum(total_loss_test) / len(total_loss_test)}")
                #loss.backward()
                #optimizer.step()
                #optimizer.zero_grad()
                progress.update(task ,completed=float(idx))
            
            #traced = th.jit.trace(
            #    lambda X: model(X),
            #    th.tensor(
            #        [encode_text("Argentina es")],
            #        dtype=th.long
            #    ).to(device),
            #    strict=False
            #)

        print(f"Epoch {epoch}, Error: {sum(total_loss_test) / len(total_loss_test)}")


        #writer.add_graph(
         #   traced,
         #   th.tensor(
         #       [encode_text("Argentina es")],
          #      dtype=th.long
          #  ).to(device)
        #)

        #writer.close()

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

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_loss["Epoch"],
                y=df_loss["Loss Train"],
                mode='lines+markers',
                name="Training Loss",
                line=dict(color='blue')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_loss["Epoch"],
                y=df_loss["Loss Test"],
                mode="lines+markers",
                name="Validation Loss",
                line=dict(color='red')
            )
        )

        fig.update_layout(
            title="Pérdida de Entrenamiento y Validación",
            xaxis_title="Época",
            yaxis_title="Pérdida",
            legend_title="Tipo de Pérdida",
            template="plotly_dark"
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
            
            last_logits = out[:, -1, :]
            
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