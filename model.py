"""
model.py

Physics-Guided Auxiliary Learning framework.

Author: JunYoung Park and Myung-Kyu Yi
"""


from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """1D Depthwise Separable Convolution block."""
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
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block using parallel depthwise separable convolutions."""
    def __init__(
        self, 
        channels: int, 
        dilation: int, 
        dropout: float,
        kernel_sizes: Sequence[int] = (3, 7), 
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = ((k - 1) * dilation) // 2
            self.branches.append(
                nn.Sequential(
                    DepthwiseSeparableConv1d(channels, channels, k, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.fusion = nn.Conv1d(channels * len(kernel_sizes), channels, 1)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        return self.fusion(torch.cat(outs, dim=1))


class NeXtTCNBlock(nn.Module):
    """Dilated Residual NeXt-TCN Block."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation: int, 
        dropout: float,
        kernel_sizes: Sequence[int] = (3, 7), 
    ):
        super().__init__()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.multi = MultiScaleConvBlock(
            channels=out_channels, 
            kernel_sizes=kernel_sizes, 
            dilation=dilation, 
            dropout=dropout
        )

        max_k = max(kernel_sizes)
        padding = ((max_k - 1) * dilation) // 2
        self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, max_k, dilation=dilation, padding=padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        residual = x

        out = self.multi(x)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.drop2(out)
        return F.gelu(out + residual)


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation (SE) Block."""
    def __init__(
        self, 
        channels: int, 
        reduction: int = 5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        b, c, _ = x.size()
        s = F.adaptive_avg_pool1d(x, 1).view(b, c)
        e = torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).view(b, c, 1)
        return x * e


class LargeKernelConv1d(nn.Module):
    """Large-Depthwise-Seperable Kernel Global Context Module."""
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 19,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.bn(self.dw(x))


class BaseNeXtTCNBackbone(nn.Module):
    """Multi-Scale NeXt-TCN Backbone."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        n_layers: int, 
        n_classes: int,
        dropout: float, 
        kernel_sizes: Sequence[int] = (3, 7), 
        large_kernel: int = 19, 
        use_se: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.lk1 = LargeKernelConv1d(hidden_dim, kernel_size=large_kernel)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                NeXtTCNBlock(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    kernel_sizes=kernel_sizes, 
                    dilation=2 ** i, 
                    dropout=dropout
                )
            )

        self.lk2 = LargeKernelConv1d(hidden_dim, kernel_size=large_kernel)
        self.use_se = use_se
        self.se = SqueezeExcitation1d(hidden_dim) if use_se else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def extract_seq_feat(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)           
        x = self.input_proj(x)          
        x = F.gelu(self.lk1(x))
        for blk in self.blocks:
            x = blk(x)
        x = F.gelu(self.lk2(x))
        x = self.se(x)
        return x                        

    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.extract_seq_feat(x)
        pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        pooled = self.norm(pooled)
        return self.head(pooled)


class PhysRegNeXtNet(BaseNeXtTCNBackbone):
    """PhysReg-NeXt Architecture."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        n_layers: int, 
        n_classes: int,
        dropout: float,
        kernel_sizes: Sequence[int] = (3, 7), 
        large_kernel: int = 19, 
        use_se: bool = True,
    ):
        super().__init__(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers, 
            n_classes=n_classes,
            kernel_sizes=kernel_sizes, 
            large_kernel=large_kernel, 
            dropout=dropout, 
            use_se=use_se
        )
        
        c = self.head.in_features
        self.gravity_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(),
            nn.Linear(c // 2, 3),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        return_gravity: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        feat = self.extract_seq_feat(x)                 
        pooled = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        
        if not return_gravity:
            return logits
            
        gvec = self.gravity_head(feat.transpose(1, 2))  
        return logits, gvec
