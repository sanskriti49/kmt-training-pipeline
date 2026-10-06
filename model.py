"""
model.py - 1D-CNN + BiGRU Siamese Multimodal Metric Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    def __init__(self, in_features: int, conv_channels: int, rnn_hidden: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(rnn_hidden * 2, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        gru_out, _ = self.gru(x_conv)
        pooled = torch.mean(gru_out, dim=1)
        return F.relu(self.fc(pooled))


class MultimodalBiometricEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.key_branch = TemporalBranch(in_features=4, conv_channels=32, rnn_hidden=64, kernel_size=3)
        self.mouse_branch = TemporalBranch(in_features=5, conv_channels=32, rnn_hidden=64, kernel_size=5)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, keys: torch.Tensor, mouse: torch.Tensor) -> torch.Tensor:
        e_key = self.key_branch(keys)
        e_mouse = self.mouse_branch(mouse)
        fused = torch.cat([e_key, e_mouse], dim=-1)
        z = self.fusion(fused)
        return F.normalize(z, p=2, dim=-1)