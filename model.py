import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


"""
    - Physically Regularized Temporal Convolutional Network (PhysReg-TCN)
    - Author: JunYoungPark and Myung-Kyu Yi
"""


class DepthwiseSeparableConv1d(nn.Module):
    """1D depthwise separable convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiScaleConvBlock(nn.Module):
    """Multi-scale temporal convolution block with depthwise separable convs"""
    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int] = [3, 5, 7],
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            padding = ((k - 1) * dilation) // 2
            branch = nn.ModuleDict(
                {
                    "conv": DepthwiseSeparableConv1d(
                        channels,
                        channels,
                        kernel_size=k,
                        dilation=dilation,
                        padding=padding,
                    ),
                    "norm": nn.BatchNorm1d(channels),
                    "dropout": nn.Dropout(dropout),
                }
            )
            self.branches.append(branch)

        self.fusion = nn.Conv1d(
            in_channels=channels * len(kernel_sizes),
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        target_length = x.size(2)

        for branch in self.branches:
            out = branch["conv"](x)
            if out.size(2) != target_length: 
                out = out[:, :, :target_length]
            out = branch["norm"](out)
            out = F.gelu(out)
            out = branch["dropout"](out)
            outputs.append(out)

        multi_scale = torch.cat(outputs, dim=1)
        return self.fusion(multi_scale)  


class ModernTCNBlock(nn.Module):
    """Modern TCN residual block"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int] = [3, 7],
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.multi_conv1 = MultiScaleConvBlock(
            channels=out_channels,
            kernel_sizes=kernel_sizes,
            dilation=dilation,
            dropout=dropout,
        )

        max_k = max(kernel_sizes) if isinstance(kernel_sizes, list) else kernel_sizes
        padding = ((max_k - 1) * dilation) // 2

        self.conv2 = DepthwiseSeparableConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=max_k,
            dilation=dilation,
            padding=padding,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        target_length = x.size(2)

        if self.downsample is not None:
            x = self.downsample(x)  
            residual = x

        out = self.multi_conv1(x)   
        if out.size(2) != target_length:
            out = out[:, :, :target_length]

        out = self.conv2(out)      
        if out.size(2) != target_length:
            out = out[:, :, :target_length]

        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout2(out)

        return F.gelu(out + residual)


class SqueezeExcitation1d(nn.Module):
    """1D Squeeze-and-Excitation (SE) block"""
    def __init__(self, channels: int, reduction: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()
        squeeze = F.adaptive_avg_pool1d(x, 1).view(batch, channels)  
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch, channels, 1)
        return x * excitation


class LargeKernelConv1d(nn.Module):
    """Depthwise 1D convolution with a large kernel to capture long-range patterns"""
    def __init__(self, channels: int, kernel_size: int = 21):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.norm(out)
        return out


class BaseModernTCNHAR(nn.Module):
    """Base Modern TCN backbone for sensor-based HAR"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_classes: int,
        kernel_sizes: list[int],
        large_kernel: int,
        dropout: float,
        use_se: bool,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            bias=False,
        )

        self.large_kernel_conv = LargeKernelConv1d(
            channels=hidden_dim,
            kernel_size=large_kernel,
        )

        self.tcn_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                ModernTCNBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.final_large_kernel = LargeKernelConv1d(
            channels=hidden_dim,
            kernel_size=large_kernel,
        )

        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation1d(channels=hidden_dim)

        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = self.input_proj(x)        
        x = self.large_kernel_conv(x)
        x = F.gelu(x)

        for block in self.tcn_blocks:
            x = block(x)

        x = self.final_large_kernel(x)
        x = F.gelu(x)

        if self.use_se:
            x = self.se(x)

        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        x = self.norm_final(x)
        logits = self.head(x)
        return logits


class PhysicsModernTCNHAR(BaseModernTCNHAR):
    """Physics-guided Modern TCN for HAR"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.head.in_features

        self.gravity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_gravity: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)

        feat = self.input_proj(x)      
        feat = self.large_kernel_conv(feat)
        feat = F.gelu(feat)

        for block in self.tcn_blocks:
            feat = block(feat)

        feat = self.final_large_kernel(feat)
        feat = F.gelu(feat)

        if self.use_se:
            feat = self.se(feat)

        pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  
        pooled = self.norm_final(pooled)
        logits = self.head(pooled)

        if not return_gravity:
            return logits

        seq_feat = feat.transpose(1, 2).contiguous()
        g_vec = self.gravity_head(seq_feat) 

        return logits, g_vec
