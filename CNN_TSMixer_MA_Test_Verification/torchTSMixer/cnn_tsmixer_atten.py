from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature

class TSMixerWithCNNAndAttention(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

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
        num_attention_heads: int = 5,
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_channels = input_channels

        activation_fn = getattr(F, activation_fn)
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # CNN 前处理层（处理 time-dim）
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=cnn_kernel_size, padding="same")

        # Mixer 主体
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

        # 将序列长度投影到预测长度（这里预测长度应该是1）
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)

        # 多头注意力，注意：query, key, value 维度必须相同
        # 这里embed_dim应该是output_channels，因为时间步已经投影到prediction_length
        self.attn = nn.MultiheadAttention(embed_dim=output_channels, num_heads=num_attention_heads, batch_first=True)
        self.output_proj = nn.Linear(output_channels, output_channels)
        
        # 最终输出层：从output_channels映射到10个输出值
        # 因为prediction_length=1，所以输入维度是output_channels，输出维度是10
        self.final_output = nn.Sequential(
            nn.Linear(output_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 10)  # 输出10个值
        )

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
        # x: [B, T, C] = [B, 60, 3]
        batch_size = x.shape[0]
        
        # CNN: convert to [B, C, T] → back to [B, T, C]
        x = x.permute(0, 2, 1)  # [B, 3, 60]
        x = self.conv1(x)  # [B, 3, 60]
        x = x.permute(0, 2, 1)  # [B, 60, 3]

        # Mixer
        x = self.mixer_layers(x)  # [B, 60, output_channels]

        # Temporal projection: 从60个时间步投影到1个时间步
        x_temp = feature_to_time(x)  # [B, output_channels, 60]
        x_temp = self.temporal_projection(x_temp)  # [B, output_channels, 1]
        x = time_to_feature(x_temp)  # [B, 1, output_channels]

        # Multihead attention (on pred_len=1)
        # 注意：当pred_len=1时，注意力机制实际上退化成了线性变换
        attn_out, _ = self.attn(x, x, x)  # [B, 1, output_channels]
        x = self.output_proj(attn_out)  # [B, 1, output_channels]
        
        # 取最后一个时间步（这里实际上只有一个时间步）
        x = x[:, -1, :]  # [B, output_channels]
        
        # 通过最终输出层得到10个值
        x = self.final_output(x)  # [B, 10]
        
        return x


if __name__ == "__main__":
    # 测试模型
    m = TSMixerWithCNNAndAttention(
        sequence_length=60, 
        prediction_length=1,  # 输出1个时间步
        input_channels=3, 
        output_channels=16,   # 可以调整这个值
        num_blocks=5,
        dropout_rate=0.2,
        ff_dim=256,
        num_attention_heads=5
    )
    
    # 模拟输入：batch_size=4, 60个时间步, 3个特征
    x = torch.randn(4, 60, 3)
    y = m(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")  # 应该是 [4, 10]