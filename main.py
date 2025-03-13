import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import pandas as pd

import plotly.express as px

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from pathlib import Path

import itertools

from utils import get_data, decode_text, encode_text

class WikiDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = get_data()

    def __len__(self):
        return len(self.data)

    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return (th.tensor(self.data.select("word")\
            .to_numpy())[idx],
                th.tensor(self.data.select("prediction")\
                          .to_numpy())[idx])

    @property
    def words(self):
        plane_words = list(itertools.chain(*self.data.select("word").to_numpy().tolist()))
        return plane_words

    @property
    def labels(self):
        plane_words = list(itertools.chain(*self.data.select("prediction").to_numpy().tolist()))
        return plane_words

    @property
    def get_words(self):
        return decode_text(self.words)


class WordRRN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, num_steps, pad_token=0):
        super(WordRRN, self).__init__()

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

dataset = WikiDataset()

train_dataset, test_dataset = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = max(dataset.words)+1
emb_size = 128
output_size = vocab_size
hidden_size = 256
num_steps = 3

num_epochs = 5

model = WordRRN(vocab_size, emb_size, hidden_size, output_size, num_steps, pad_token=0)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss = []

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
                try:
                    out = model(input_words)
                except Exception as e:
                    print("Error:", e)
                    print("Word:", input_words)
                    continue

                loss = criterion(out, target.squeeze(dim=1))
                progress.update(task, loss=f"{loss.item():.4f}")
                total_loss.append(loss.item())
                progress.update(task, loss_mean=f"{sum(total_loss)/len(total_loss)}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress.update(task ,completed=float(idx))

        model.eval()

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

                try:
                    out = model(input_words)
                except Exception as e:
                    print("Error:", e)
                    print("Word:", input_words)
                    continue

            loss = criterion(out, target.squeeze(dim=1))
            progress.update(task, loss=f"{loss.item():.4f}")
            total_loss.append(loss.item())
            progress.update(task, loss_mean=f"{sum(total_loss)/len(total_loss)}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.update(task ,completed=float(idx))

        print(f"Epoch {epoch}, Error: {sum(total_loss)/len(total_loss)}")