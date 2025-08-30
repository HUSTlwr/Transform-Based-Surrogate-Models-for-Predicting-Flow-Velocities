from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature


class TSMixer(nn.Module):
    

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
    ):
        super().__init__()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
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

        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x


if __name__ == "__main__":
    m = TSMixer(10, 5, 2, output_channels=4)
    x = torch.randn(3, 10, 2)
    y = m(x)
