from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature
from kan import *

class TSMixerWithCNNAndAttention(nn.Module):
    

    

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
        cnn_kernel_size: int = 3,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        activation_fn = getattr(F, activation_fn)
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm


        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=cnn_kernel_size, padding="same")


        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        self.temporal_projection = nn.Linear(sequence_length, prediction_length)
        
        self.attn = nn.MultiheadAttention(embed_dim=output_channels, num_heads=num_attention_heads, batch_first=True)
        self.output_proj = nn.Linear(output_channels, output_channels)
        self.kan=KAN(width=[5,5,5], grid=5, k=3) 

    def _build_mixer(self, num_blocks, input_channels, output_channels, **kwargs):
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]
        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        # CNN: convert to [B, C, T] → back to [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)

        # Mixer
        x = self.mixer_layers(x)

        # Temporal projection
        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)  # [B, pred_len, C]

        # Multihead attention (on pred_len)
        attn_out, _ = self.attn(x, x, x)
        x = self.output_proj(attn_out)


        x = x[:, -1, :]                    
        pred = self.kan(x)                  
        return pred #x


if __name__ == "__main__":
    m = TSMixer(10, 5, 2, output_channels=4)
    x = torch.randn(3, 10, 2)
    y = m(x)
