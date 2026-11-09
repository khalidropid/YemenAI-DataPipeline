from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _sequence_lengths(x: torch.Tensor) -> torch.Tensor:
    """Estimate valid sequence lengths from zero-padded [B,T,F] features."""
    valid = x.abs().sum(dim=-1).gt(1e-8)
    lengths = valid.sum(dim=1).clamp_min(1)
    return lengths


class MLPClassifier(nn.Module):
    """MLP for MFCC sequences using temporal mean and standard deviation."""
    def __init__(self, num_classes: int, input_dim: int = 40, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MLP expects [batch,time,features], got {tuple(x.shape)}")
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return self.net(torch.cat((mean, std), dim=1))


class CNNClassifier(nn.Module):
    """2-D CNN for log-Mel spectrograms of arbitrary time length."""
    def __init__(self, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128, 96), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CNN expects [batch,1,mel,time], got {tuple(x.shape)}")
        return self.classifier(self.features(x))


class RecurrentClassifier(nn.Module):
    """LSTM/GRU classifier for padded MFCC sequences."""
    def __init__(self, num_classes: int, cell: str = "lstm",
                 input_dim: int = 40, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RNN expects [batch,time,features], got {tuple(x.shape)}")
        lengths = _sequence_lengths(x).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        steps = torch.arange(out.size(1), device=out.device)[None, :]
        mask = steps < lengths.to(out.device)[:, None]
        scores = self.attention(out).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attended = (out * weights).sum(dim=1)
        masked = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        mean = masked.sum(dim=1) / lengths.to(out.device).clamp_min(1).unsqueeze(1)
        return self.classifier(torch.cat((attended, mean), dim=1))


class CNNBiLSTMAttention(nn.Module):
    """CNN feature extractor followed by a bidirectional LSTM and attention."""
    def __init__(self, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.rnn = nn.LSTM(
            input_size=64 * 16, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CNN-BiLSTM expects [batch,1,mel,time], got {tuple(x.shape)}")
        z = self.cnn(x)                         # [B,64,16,T']
        z = z.mean(dim=2).transpose(1, 2)      # [B,T',64]
        out, _ = self.rnn(z)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        attended = (out * weights.unsqueeze(-1)).sum(dim=1)
        mean = out.mean(dim=1)
        return self.head(torch.cat((attended, mean), dim=1))


def build_model(name: str, num_classes: int = 5) -> nn.Module:
    """Construct one of the model architectures exposed by the training CLI."""
    key = name.lower()
    if key == "mlp":
        return MLPClassifier(num_classes)
    if key == "cnn":
        return CNNClassifier(num_classes)
    if key == "lstm":
        return RecurrentClassifier(num_classes, cell="lstm")
    if key == "gru":
        return RecurrentClassifier(num_classes, cell="gru")
    if key == "cnn_bilstm_attention":
        return CNNBiLSTMAttention(num_classes)
    raise ValueError(
        f"Unknown model '{name}'. Choose: mlp, cnn, lstm, gru, cnn_bilstm_attention."
    )
