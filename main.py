import torch as th
from pygments.lexer import words
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from sklearn.decomposition import PCA

import pandas as pd

import plotly.express as px

from rich.progress import track

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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

vocab_size = max(dataset.words)+1
emb_size = 128
output_size = vocab_size
hidden_size = 256
num_steps = 3

num_epochs = 1

model = WordRRN(vocab_size, emb_size, hidden_size, output_size, num_steps, pad_token=0)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    print(f"Max Indexing: {max(dataset.words)}")

    for param, value in model.state_dict().items():
        print(param, value.shape)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in track(enumerate(dataloader), description="Batch...", total=len(dataloader)):
            out = "XD"

            input_words = batch[0]
            target = batch[1]

            optimizer.zero_grad()
            try:
                out = model(input_words)
            except Exception as e:
                print(input_words)
                print(e)
                continue
            #print("target shape: ", target.shape, "\n", "out shape: ", out.shape)
            loss = criterion(out, target.squeeze())

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')