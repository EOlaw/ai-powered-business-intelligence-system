"""
InsightSerenity AI Engine — GRU
=================================
Gated Recurrent Unit (Cho et al., 2014).

The GRU simplifies the LSTM by merging the cell state and hidden state into a
single hidden state h_t, and replacing the three LSTM gates with two gates:

    Reset gate: how much of the previous hidden state to use
        r_t = sigmoid(W_r @ [h_{t-1}, x_t])

    Update gate: how much to update the hidden state vs keep the old one
        z_t = sigmoid(W_z @ [h_{t-1}, x_t])

    Candidate hidden state (reset gate applied):
        n_t = tanh(W_n @ x_t + r_t ⊗ (U_n @ h_{t-1}))

    New hidden state:
        h_t = (1 - z_t) ⊗ h_{t-1} + z_t ⊗ n_t

The update gate z_t controls the balance between:
    - z_t ≈ 0: keep old hidden state (identity, no update — long-term memory)
    - z_t ≈ 1: fully replace with new candidate (forget old state)

The reset gate r_t controls how much past context influences the new candidate:
    - r_t ≈ 0: ignore past hidden state (start fresh)
    - r_t ≈ 1: fully use past hidden state

GRU vs LSTM:
    - GRU has fewer parameters (2 gates vs 3 + cell state)
    - GRU is slightly faster to train and often performs comparably to LSTM
    - LSTM has more expressive memory control due to the separate cell state
    - For most sequence tasks, GRU and LSTM perform similarly
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GRUCell(nn.Module):
    """
    Single-step GRU cell built from scratch.

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

        # Reset and update gates — computed jointly for efficiency
        # Output: 2 × hidden_size (r and z gates)
        self.W_rz_ih = nn.Linear(input_size,  2 * hidden_size, bias=bias)
        self.W_rz_hh = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        # Candidate hidden state — split because reset gate is applied to h
        self.W_n_ih  = nn.Linear(input_size,  hidden_size, bias=bias)
        self.W_n_hh  = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        for linear in [self.W_rz_ih, self.W_n_ih]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        for linear in [self.W_rz_hh, self.W_n_hh]:
            nn.init.orthogonal_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute one GRU step.

        Args:
            x_t:    Input at current time step. Shape: (B, input_size)
            h_prev: Previous hidden state.       Shape: (B, hidden_size)

        Returns:
            h_t: New hidden state. Shape: (B, hidden_size)
        """
        H = self.hidden_size

        # Reset and update gates (computed jointly)
        gates_rz = self.W_rz_ih(x_t) + self.W_rz_hh(h_prev)   # (B, 2H)
        r_gate   = torch.sigmoid(gates_rz[:, :H])               # (B, H) — reset
        z_gate   = torch.sigmoid(gates_rz[:, H:])               # (B, H) — update

        # Candidate hidden state (reset gate applied to previous hidden state)
        n_t = torch.tanh(self.W_n_ih(x_t) + r_gate * self.W_n_hh(h_prev))

        # Interpolate between old and candidate hidden states
        h_t = (1.0 - z_gate) * h_prev + z_gate * n_t

        return h_t


class GRU(nn.Module):
    """
    Multi-layer, optionally bidirectional GRU over a full sequence.

    Args:
        input_size:    Input feature dimension.
        hidden_size:   Hidden state dimension.
        num_layers:    Stacked GRU layers. Default 1.
        bias:          Add bias terms. Default True.
        dropout:       Inter-layer dropout. Default 0.0.
        bidirectional: Process sequence both ways. Default False.
        batch_first:   If True, expect (B, T, D) input.
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

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.batch_first   = batch_first
        self.num_directions = 2 if bidirectional else 1

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input = input_size if layer == 0 else hidden_size * self.num_directions
            self.cells.append(GRUCell(layer_input, hidden_size, bias))
            if bidirectional:
                self.cells.append(GRUCell(layer_input, hidden_size, bias))

        self.dropout = nn.Dropout(p=dropout) if (dropout > 0.0 and num_layers > 1) else None

    def forward(
        self,
        x: Tensor,
        h_0: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Process a full sequence through all GRU layers.

        Args:
            x:   Input sequence.
                   batch_first=True:  (B, T, input_size)
                   batch_first=False: (T, B, input_size)
            h_0: Initial hidden state.
                   (num_layers × num_directions, B, hidden_size)
                   If None, zeros.

        Returns:
            Tuple of:
                output: (B, T, hidden_size × num_directions) [if batch_first]
                h_n:    (num_layers × num_directions, B, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        T, B, _ = x.shape

        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions, B, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        all_h: list = []
        layer_input = x

        for layer in range(self.num_layers):
            fwd_idx  = layer * self.num_directions
            h_fwd    = h_0[fwd_idx]
            fwd_cell = self.cells[fwd_idx]

            fwd_outputs = []
            for t in range(T):
                h_fwd = fwd_cell(layer_input[t], h_fwd)
                fwd_outputs.append(h_fwd)
            fwd_stack = torch.stack(fwd_outputs, dim=0)

            if self.bidirectional:
                bwd_idx  = fwd_idx + 1
                h_bwd    = h_0[bwd_idx]
                bwd_cell = self.cells[bwd_idx]
                bwd_outputs = []
                for t in reversed(range(T)):
                    h_bwd = bwd_cell(layer_input[t], h_bwd)
                    bwd_outputs.append(h_bwd)
                bwd_stack = torch.stack(bwd_outputs[::-1], dim=0)
                layer_out = torch.cat([fwd_stack, bwd_stack], dim=-1)
                all_h.extend([h_fwd, h_bwd])
            else:
                layer_out = fwd_stack
                all_h.append(h_fwd)

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_out = self.dropout(layer_out)

            layer_input = layer_out

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)

        h_n = torch.stack(all_h, dim=0)
        return output, h_n

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}"
        )
