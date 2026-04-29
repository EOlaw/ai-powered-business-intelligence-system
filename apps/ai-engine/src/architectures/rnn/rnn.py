"""
InsightSerenity AI Engine — Vanilla RNN
========================================
The simplest recurrent architecture. A Recurrent Neural Network maintains
a hidden state h_t that is updated at each time step based on the current
input x_t and the previous hidden state h_{t-1}.

Update equation:
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)

This makes the model order-dependent — position i knows about all previous
positions through the hidden state chain. This is fundamentally different
from attention, which is position-agnostic by default.

Limitation: the vanilla RNN suffers from vanishing/exploding gradients
for long sequences because the gradient of h_t with respect to h_0 requires
multiplying the same weight matrix T times. For this reason:
    - For short sequences: use vanilla RNN
    - For medium sequences: use LSTM or GRU (both in this package)
    - For long sequences: use Transformers

This implementation builds the RNN from scratch (no nn.RNN wrapper)
to expose the internal mechanics clearly, which is educational and also
allows us to modify the update rule in the LSTM/GRU implementations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class VanillaRNNCell(nn.Module):
    """
    Single step of a Vanilla RNN.

    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

    Args:
        input_size:  Dimension of input x_t.
        hidden_size: Dimension of hidden state h_t.
        bias:        Add bias terms. Default True.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Input → hidden
        self.W_ih = nn.Linear(input_size, hidden_size, bias=bias)
        # Hidden → hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Orthogonal init for W_hh — preserves gradient magnitude over time
        nn.init.orthogonal_(self.W_hh.weight)
        nn.init.xavier_uniform_(self.W_ih.weight)
        if bias:
            nn.init.zeros_(self.W_ih.bias)
            nn.init.zeros_(self.W_hh.bias)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute one RNN step.

        Args:
            x_t:    Input at current time step.  Shape: (B, input_size)
            h_prev: Previous hidden state.       Shape: (B, hidden_size)

        Returns:
            h_t: New hidden state. Shape: (B, hidden_size)
        """
        h_t = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        eps = torch.finfo(h_t.dtype).eps
        return h_t.clamp(min=-1.0 + eps, max=1.0 - eps)


class VanillaRNN(nn.Module):
    """
    Multi-layer, optionally bidirectional Vanilla RNN over a full sequence.

    Unrolls the RNNCell across all time steps and returns all hidden states
    (sequence of outputs) and the final hidden state.

    Args:
        input_size:  Input dimension.
        hidden_size: Hidden state dimension.
        num_layers:  Number of stacked RNN layers. Default 1.
        bias:        Use bias terms. Default True.
        dropout:     Dropout applied between layers (not on last layer).
        bidirectional: If True, process sequence in both directions and
                       concatenate hidden states. Output dim = 2 × hidden_size.
        batch_first: If True, input shape is (B, T, D). If False, (T, B, D).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.batch_first  = batch_first
        self.num_directions = 2 if bidirectional else 1

        # Build one cell per layer per direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            # Forward direction
            layer_input = input_size if layer == 0 else hidden_size * self.num_directions
            self.cells.append(VanillaRNNCell(layer_input, hidden_size, bias))

            if bidirectional:
                # Backward direction
                self.cells.append(VanillaRNNCell(layer_input, hidden_size, bias))

        self.dropout = nn.Dropout(p=dropout) if (dropout > 0.0 and num_layers > 1) else None

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process a full sequence through all layers.

        Args:
            x:   Input tensor.
                   batch_first=True:  (B, T, input_size)
                   batch_first=False: (T, B, input_size)
            h_0: Initial hidden state.
                   Shape: (num_layers * num_directions, B, hidden_size)
                   If None, initialised to zeros.

        Returns:
            Tuple of:
                output: All hidden states at final layer.
                        Shape: (B, T, hidden_size * num_directions) [batch_first]
                h_n:    Final hidden state for all layers and directions.
                        Shape: (num_layers * num_directions, B, hidden_size)
        """
        # Normalise to (T, B, D)
        if self.batch_first:
            x = x.transpose(0, 1)   # (T, B, D)

        T, B, _ = x.shape

        # Initialise hidden states
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                B,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        all_final_hidden: list = []
        layer_input = x   # Start with the raw input

        for layer in range(self.num_layers):
            fwd_cell_idx = layer * self.num_directions
            h_fwd = h_0[fwd_cell_idx]   # (B, hidden_size)

            # Forward pass
            fwd_cell = self.cells[fwd_cell_idx]
            fwd_outputs = []
            for t in range(T):
                h_fwd = fwd_cell(layer_input[t], h_fwd)
                fwd_outputs.append(h_fwd)
            fwd_stack = torch.stack(fwd_outputs, dim=0)   # (T, B, H)

            if self.bidirectional:
                bwd_cell_idx = fwd_cell_idx + 1
                h_bwd = h_0[bwd_cell_idx]
                bwd_cell = self.cells[bwd_cell_idx]
                bwd_outputs = []
                for t in reversed(range(T)):
                    h_bwd = bwd_cell(layer_input[t], h_bwd)
                    bwd_outputs.append(h_bwd)
                bwd_stack = torch.stack(bwd_outputs[::-1], dim=0)  # (T, B, H)

                layer_output = torch.cat([fwd_stack, bwd_stack], dim=-1)  # (T, B, 2H)
                all_final_hidden.extend([h_fwd, h_bwd])
            else:
                layer_output = fwd_stack
                all_final_hidden.append(h_fwd)

            # Apply inter-layer dropout (not on the last layer)
            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        # output: (T, B, H*directions) → (B, T, H*directions) if batch_first
        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)

        # h_n: (num_layers * num_directions, B, H)
        h_n = torch.stack(all_final_hidden, dim=0)

        return output, h_n

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}"
        )
