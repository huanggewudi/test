from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / sqrt(dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 1, -float("inf"))

    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores, v)


class Head(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.q_linear = nn.Linear(dim, head_dim)
        self.k_linear = nn.Linear(dim, head_dim)
        self.v_linear = nn.Linear(dim, head_dim)

    def forward(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        return attention(q, k, v)


class MultiHead(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        head_dim = dim // num_heads
        self.heads = nn.ModuleList(
            [Head(dim, head_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        output = torch.cat([head(q, k, v) for head in self.heads], dim=-1)
        return self.linear(output)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.multi_head = MultiHead(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        q = x
        k = x
        v = x

        hidden = self.multi_head(q, k, v)
        x = hidden + x
        x = self.ln1(x)

        hidden = self.feed_forward(x)
        x = x + hidden
        x = self.ln2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_ids):
        embs = self.emb(input_ids)
        embs = self.ln(embs)
        return embs


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads):
        super().__init__()
        self.emb = Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, input_ids):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    vocab_size = 1000
    dim = 512
    num_layers = 6
    num_heads = 8

    batch_size = 32
    seq_length = 50

    encoder = Encoder(vocab_size, dim, num_layers, num_heads)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    output = encoder(input_ids)

    print("Output shape:", output.shape)
    print(output)
