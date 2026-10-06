"""model_v2.py - Masked-pooling multimodal encoder (Exp03+).

Fixes the two strongest architectural weaknesses audited in model.py:
  1. UNMASKED MEAN POOLING: TemporalBranch did torch.mean(gru_out, dim=1)
     over zero-padded rows, so short trials were pulled toward the padding
     origin and embeddings carried length-dependent bias. Here pooling is
     masked by true sequence lengths.
  2. BATCH-SIZE-DEPENDENT FUSION: BatchNorm1d in the fusion head depends on
     batch statistics (noisy for small triplet batches, frozen running stats
     at few-shot eval). Replaced with LayerNorm, which is batch-independent.

Keeps the same interface dims (64-D per branch, 128-D L2-normalized output)
and adds a light modality gate so one dominant modality cannot wash out
the other.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTemporalBranch(nn.Module):
    def __init__(self, in_features: int, conv_channels: int, rnn_hidden: int,
                 kernel_size: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, conv_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=rnn_hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden * 2, 64)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), lengths: (B,) true lengths BEFORE padding
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
        gru_out, _ = self.gru(x_conv)                          # (B, T, 2H)
        T = gru_out.size(1)
        lengths = lengths.clamp(min=1, max=T)
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)).float().unsqueeze(-1)   # (B, T, 1)
        pooled = (gru_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.relu(self.fc(pooled))


class MultimodalBiometricEncoderV2(nn.Module):
    def __init__(self, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.key_branch = MaskedTemporalBranch(in_features=4, conv_channels=32,
                                               rnn_hidden=64, kernel_size=3,
                                               dropout=dropout)
        self.mouse_branch = MaskedTemporalBranch(in_features=5, conv_channels=32,
                                                 rnn_hidden=64, kernel_size=5,
                                                 dropout=dropout)
        # Modality gate: lets the network down-weight a noisy modality per trial
        self.gate = nn.Sequential(nn.Linear(128, 2), nn.Softmax(dim=-1))
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, keys: torch.Tensor, mouse: torch.Tensor,
                key_len: torch.Tensor = None,
                mouse_len: torch.Tensor = None) -> torch.Tensor:
        if key_len is None:
            key_len = torch.full((keys.size(0),), keys.size(1),
                                 dtype=torch.long, device=keys.device)
        if mouse_len is None:
            mouse_len = torch.full((mouse.size(0),), mouse.size(1),
                                   dtype=torch.long, device=mouse.device)
        e_key = self.key_branch(keys, key_len)
        e_mouse = self.mouse_branch(mouse, mouse_len)
        w = self.gate(torch.cat([e_key, e_mouse], dim=-1))  # (B, 2)
        fused = torch.cat([w[:, :1] * e_key, w[:, 1:] * e_mouse], dim=-1)
        return F.normalize(self.fusion(fused), p=2, dim=-1)
