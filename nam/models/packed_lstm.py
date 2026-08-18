# File: packed_lstm.py
# Created Date: 2024
# Author: Neural Amp Modeler Contributors

"""
Packed LSTM model with dilated convolution preprocessing layers.

Combines multiple dilated convolutional layers with LSTM for improved temporal modeling.
"""

from typing import Optional as _Optional, Dict as _Dict, Any as _Any, List as _List
import numpy as _np
import torch as _torch
import torch.nn as _nn

from ._abc import ImportsWeights as _ImportsWeights
from .base import BaseNet as _BaseNet


class _DilatedConvBlock(_nn.Module):
    """Single dilated convolution block with activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = _nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
            bias=True,
        )
        if activation == "LeakyReLU":
            self.activation = _nn.LeakyReLU()
        elif activation == "ReLU":
            self.activation = _nn.ReLU()
        else:
            self.activation = _nn.Identity()

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        x = self.conv(x)
        # Remove padding added by dilation to maintain output length
        x = x[:, :, : -(max(self.conv.padding) - 1)] if max(self.conv.padding) > 1 else x
        return self.activation(x)


class _PackedLSTMCore(_nn.Module):
    """Core LSTM with learnable initial state."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = _nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._initial_hidden = _nn.Parameter(
            _torch.zeros((num_layers, hidden_size))
        )
        self._initial_cell = _nn.Parameter(
            _torch.zeros((num_layers, hidden_size))
        )

    def forward(
        self,
        x: _torch.Tensor,
        hidden_state: _Optional[tuple] = None,
    ) -> tuple:
        if hidden_state is None:
            batch_size = x.shape[0]
            h = _torch.tile(self._initial_hidden[:, None, :], (1, batch_size, 1))
            c = _torch.tile(self._initial_cell[:, None, :], (1, batch_size, 1))
            hidden_state = (h, c)
        return self.lstm(x, hidden_state)


class PackedLSTM(_BaseNet, _ImportsWeights):
    """
    PackedLSTM model: dilated convolutions + LSTM.

    The dilated convolutions provide multi-scale temporal feature extraction,
    while the LSTM learns the sequential dependencies.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 2,
        conv_layers_config: _Optional[_List[_Dict[str, _Any]]] = None,
        dropout: float = 0.1,
        head_kernel_size: int = 1,
        head_scale: float = 1.0,
        sample_rate: _Optional[float] = None,
    ):
        """
        :param input_size: Input dimension (usually 1 for mono audio)
        :param hidden_size: Hidden state size for LSTM
        :param num_layers: Number of LSTM layers
        :param conv_layers_config: List of dicts with 'kernel_size' and 'dilation' keys
        :param dropout: Dropout rate for LSTM
        :param head_kernel_size: Kernel size for final linear head
        :param head_scale: Scaling factor for head output
        :param sample_rate: Sample rate of audio
        """
        super().__init__(sample_rate=sample_rate)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._head_scale = head_scale
        self._dropout = dropout

        # Build dilated convolution layers
        conv_layers_config = conv_layers_config or []
        self._conv_layers = _nn.ModuleList()

        if conv_layers_config:
            in_channels = input_size
            for layer_config in conv_layers_config:
                kernel_size = layer_config.get("kernel_size", 6)
                dilation = layer_config.get("dilation", 1)
                out_channels = hidden_size

                conv_block = _DilatedConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    activation=layer_config.get("activation", "LeakyReLU"),
                )
                self._conv_layers.append(conv_block)
                in_channels = out_channels

            # LSTM input is hidden_size (output of last conv)
            lstm_input_size = hidden_size
        else:
            lstm_input_size = input_size

        # LSTM core
        self._lstm = _PackedLSTMCore(lstm_input_size, hidden_size, num_layers, dropout)

        # Output head: linear layer
        self._head = _nn.Linear(hidden_size, 1, bias=True)

    @property
    def input_device(self) -> _torch.device:
        return next(self.parameters()).device

    @property
    def receptive_field(self) -> int:
        # Calculate based on convolution layers
        if not self._conv_layers:
            return 1
        rf = 1
        for conv_block in self._conv_layers:
            kernel_size = conv_block.conv.kernel_size[0]
            dilation = conv_block.conv.dilation[0]
            rf += (kernel_size - 1) * dilation
        return rf

    @property
    def pad_start_default(self) -> bool:
        return True

    def _forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (B,L) or (B,L,1)
        :return: (B,L)
        """
        # Ensure 3D shape: (B, L, C)
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # Move to correct device
        x = x.to(self.input_device)

        # Apply dilated convolution layers
        for conv_block in self._conv_layers:
            # Conv1d expects (B, C, L), so transpose
            x = x.transpose(1, 2)
            x = conv_block(x)
            x = x.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self._lstm(x)

        # Apply output head
        output = self._head(lstm_out)  # (B, L, 1)
        output = output.squeeze(-1) * self._head_scale

        return output

    @classmethod
    def parse_config(cls, config: _Dict[str, _Any]) -> _Dict[str, _Any]:
        """
        Parse config from dict, extracting nested submodel configs if present.
        """
        config = config.copy()

        # Handle "submodels" structure for multi-channel models
        if "submodels" in config:
            # For now, use the first submodel's configuration
            # This supports the packed_model structure
            submodels = config.pop("submodels")
            if submodels:
                first_submodel = submodels[0]
                submodel_config = first_submodel.get("config", {})

                # Extract conv layer config
                layers_configs = submodel_config.get("layers_configs", [])
                if layers_configs:
                    layers_config = layers_configs[0]
                    kernel_sizes = layers_config.get("kernel_sizes", [])
                    dilations = layers_config.get("dilations", [])

                    conv_layers_config = [
                        {"kernel_size": k, "dilation": d}
                        for k, d in zip(kernel_sizes, dilations)
                    ]
                    config["conv_layers_config"] = conv_layers_config

                # Get head scale if present
                if "head_scale" in submodel_config:
                    config["head_scale"] = submodel_config["head_scale"]

        return config

    def import_weights(self, weights: _np.ndarray) -> None:
        """Import weights from a flat array."""
        # This is a simplified implementation
        # Full implementation would need to handle conv + lstm + head weights
        raise NotImplementedError("Weight import not yet implemented for PackedLSTM")

    def _export_config(self) -> _Dict[str, _Any]:
        conv_layers_config = []
        for conv_block in self._conv_layers:
            conv_layers_config.append({
                "kernel_size": conv_block.conv.kernel_size[0],
                "dilation": conv_block.conv.dilation[0],
            })

        return {
            "input_size": self._input_size,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "head_scale": self._head_scale,
            "conv_layers_config": conv_layers_config,
        }

    def _export_weights(self) -> _np.ndarray:
        """Export weights as a flat array."""
        # This is a simplified implementation
        raise NotImplementedError("Weight export not yet implemented for PackedLSTM")
