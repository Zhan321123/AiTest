import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
  def __init__(self, vocabSize, dModel):
    super().__init__()
    self.embed = nn.Embedding(vocabSize, dModel)
    self.dModel = dModel

  def forward(self, x):
    return self.embed(x) * math.sqrt(self.dModel)


class PositionEncoding(nn.Module):
  def __init__(self, dModel, maxLen=5000):
    super().__init__()
    pe = torch.zeros(maxLen, dModel)
    position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
    divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0) / dModel))
    pe[:, 0::2] = torch.sin(position * divTerm)
    pe[:, 1::2] = torch.cos(position * divTerm)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:, :x.size(1)]


def attention(q, k, v, mask=None, dropout=None):
  dK = q.size(-1)
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dK)

  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

  atten = torch.softmax(scores, dim=-1)

  if dropout is not None:
    atten = dropout(atten)

  return torch.matmul(atten, v), atten


class MultiHeadAttention(nn.Module):
  def __init__(self, h, dModel, dropout=0.1):
    super().__init__()
    assert dModel % h == 0
    self.dK = dModel // h
    self.h = h

    self.qLinear = nn.Linear(dModel, dModel)
    self.kLinear = nn.Linear(dModel, dModel)
    self.vLinear = nn.Linear(dModel, dModel)
    self.outLinear = nn.Linear(dModel, dModel)
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask=None):
    batchSize = q.size(0)
    transform = lambda x, linear: linear(x).view(batchSize, -1, self.h, self.dK).transpose(1, 2)
    q = transform(q, self.qLinear)
    k = transform(k, self.kLinear)
    v = transform(v, self.vLinear)

    x, _ = attention(q, k, v, mask, self.dropout)
    x = x.transpose(1, 2).contiguous().view(batchSize, -1, self.h * self.dK)

    return self.outLinear(x)


class FeedForward(nn.Module):
  def __init__(self, dModel, dFF, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dModel, dFF),
      nn.ReLU(),
      nn.Linear(dFF, dModel),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class AddNorm(nn.Module):
  def __init__(self, size, dropout=0.1):
    super().__init__()
    self.norm = nn.LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
  def __init__(self, dModel, atten, feedForward, dropout):
    super().__init__()
    self.atten = atten
    self.feedForward = feedForward
    self.subLayers = nn.ModuleList([
      AddNorm(dModel, dropout),
      AddNorm(dModel, dropout)
    ])

  def forward(self, x, mask):
    out1 = self.subLayers[0](x, lambda x: self.atten(x, x, x, mask))
    out2 = self.subLayers[1](out1, self.feedForward(out1))
    return out2


class DecoderLayer(nn.Module):
  def __init__(self, dModel, selfAtten, crossAtten, feedForward, dropout):
    super().__init__()
    self.selfAtten = selfAtten
    self.crossAtten = crossAtten
    self.feedForward = feedForward
    self.subLayers = nn.ModuleList([
      AddNorm(dModel, dropout),
      AddNorm(dModel, dropout),
      AddNorm(dModel, dropout)
    ])

  def forward(self, x, memory, srcMask, tgtMask):
    out1 = self.subLayers[0](x, lambda x: self.selfAtten(x, x, x, tgtMask))
    out2 = self.subLayers[1](out1, lambda x: self.crossAtten(x, memory, memory, srcMask))
    out3 = self.subLayers[2](out2, self.feedForward(out2))
    return out3


class Transformer(nn.Module):
  def __init__(self, srcVocab, tgtVocab, dModel=512, dFF=2048, N=6, h=8, dropout=0.1):
    super().__init__()
    self.srcEmbed = nn.Sequential(
      Embedding(srcVocab, dModel),
      PositionEncoding(dModel)
    )
    self.tgtEmbed = nn.Sequential(
      Embedding(tgtVocab, dModel),
      PositionEncoding(dModel)
    )
    atten = lambda: MultiHeadAttention(h, dModel, dropout)
    ff = lambda: FeedForward(dModel, dFF, dropout)

    self.encoder = nn.ModuleList([EncoderLayer(dModel, atten(), ff(), dropout) for _ in range(N)])
    self.decoder = nn.ModuleList([DecoderLayer(dModel, atten(), atten(), ff(), dropout) for _ in range(N)])

    self.out = nn.Linear(dModel, tgtVocab)

  def encode(self, src, srcMask):
    x = self.srcEmbed(src)
    for layer in self.encoder:
      x = layer(x, srcMask)
    return x

  def decode(self, memory, srcMask, tgt, tgtMask):
    x = self.tgtEmbed(tgt)
    for layer in self.decoder:
      x = layer(x, memory, srcMask, tgtMask)
    return x

  def forward(self, src, tgt, srcMask, tgtMask):
    memory = self.encode(src, srcMask)
    out = self.decode(memory, srcMask, tgt, tgtMask)
    return self.out(out)
