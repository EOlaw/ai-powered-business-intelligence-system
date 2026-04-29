"""
InsightSerenity AI Engine — LSTM
=================================
Long Short-Term Memory (Hochreiter & Schmidhuber, 1997).

The LSTM solves the vanishing gradient problem of vanilla RNNs by introducing
a separate cell state c_t (the "memory") controlled by three gates:

    Forget gate: how much of the old memory to erase
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)

    Input gate:  how much new information to write to memory
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)

    Cell candidate: new candidate values to potentially add to memory
        g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)

    Cell update:
        c_t = f_t ⊗ c_{t-1} + i_t ⊗ g_t

    Output gate: how much of the cell state to expose as hidden state
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊗ tanh(c_t)

The cell state c_t can flow through time with only multiplicative interactions
(the forget gate), giving gradients a path that avoids the matrix multiplication
that causes vanishing gradients in vanilla RNNs.

Key advantage over Transformer: O(T) computation vs O(T²) for attention.
Key disadvantage: cannot be parallelised over time steps.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LSTMCell(nn.Module):
    """
    Single-step LSTM cell implementing all four gates from scratch.

    All four gate projections are computed in a single batched matrix
    multiplication (4 × hidden_size output) for efficiency, then split.

    Args:
        input_size:  Dimension of input x_t.
        hidden_size: Dimension of hidden state h_t and cell state c_t.
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

        # Single combined projection: computes all 4 gates at once
        # Output: 4 × hidden_size (i, f, g, o gates in order)
        self.W_ih = nn.Linear(input_size,  4 * hidden_size, bias=bias)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise LSTM weights.

        For input weights: Xavier uniform.
        For hidden weights: Orthogonal (preserves gradient magnitude).
        Forget gate bias: initialised to 1.0 (Jozefowicz et al., 2015).
            This encourages the LSTM to remember by default at the start
            of training, which is empirically beneficial.
        """
        nn.init.xavier_uniform_(self.W_ih.weight)
        nn.init.orthogonal_(self.W_hh.weight)

        if self.W_ih.bias is not None:
            nn.init.zeros_(self.W_ih.bias)
            nn.init.zeros_(self.W_hh.bias)
            # Set forget gate bias to 1 (indices hidden_size to 2*hidden_size)
            # Gate order: i (0), f (1), g (2), o (3)
            H = self.hidden_size
            self.W_ih.bias.data[H:2*H].fill_(1.0)
            self.W_hh.bias.data[H:2*H].fill_(1.0)

    def forward(
        self,
        x_t: Tensor,
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute one LSTM step.

        Args:
            x_t:   Input at current time step. Shape: (B, input_size)
            state: Tuple (h_{t-1}, c_{t-1})
                       h_{t-1}: previous hidden state (B, hidden_size)
                       c_{t-1}: previous cell state   (B, hidden_size)

        Returns:
            Tuple (h_t, c_t):
                h_t: new hidden state (B, hidden_size)
                c_t: new cell state   (B, hidden_size)
        """
        h_prev, c_prev = state

        # Combined gate computation: (B, 4 * hidden_size)
        gates = self.W_ih(x_t) + self.W_hh(h_prev)

        H = self.hidden_size

        # Split into four gates (input, forget, cell candidate, output)
        i_gate = torch.sigmoid(gates[:, 0*H : 1*H])   # Input gate
        f_gate = torch.sigmoid(gates[:, 1*H : 2*H])   # Forget gate
        g_gate = torch.tanh(   gates[:, 2*H : 3*H])   # Cell candidate
        o_gate = torch.sigmoid(gates[:, 3*H : 4*H])   # Output gate

        # Cell state update: forget old + write new
        c_t = f_gate * c_prev + i_gate * g_gate

        # Hidden state: gated cell state
        h_t = o_gate * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    """
    Multi-layer, optionally bidirectional LSTM over a full sequence.

    Args:
        input_size:    Input feature dimension.
        hidden_size:   Hidden state dimension.
        num_layers:    Stacked LSTM layers. Default 1.
        bias:          Add bias terms. Default True.
        dropout:       Inter-layer dropout (not on final layer). Default 0.
        bidirectional: Process sequence both ways. Doubles output dim.
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
            self.cells.append(LSTMCell(layer_input, hidden_size, bias))
            if bidirectional:
                self.cells.append(LSTMCell(layer_input, hidden_size, bias))

        self.dropout = nn.Dropout(p=dropout) if (dropout > 0.0 and num_layers > 1) else None

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Process a full sequence through all LSTM layers.

        Args:
            x:     Input sequence.
                     batch_first=True:  (B, T, input_size)
                     batch_first=False: (T, B, input_size)
            state: Optional (h_0, c_0) tuple.
                     Each: (num_layers × num_directions, B, hidden_size)
                     If None, initialised to zeros.

        Returns:
            Tuple of:
                output: (B, T, hidden_size * num_directions) [batch_first]
                        All hidden states at the topmost layer.
                (h_n, c_n): Final hidden and cell states.
                        Each: (num_layers × num_directions, B, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)   # (T, B, D)

        T, B, _ = x.shape

        # Initialise states
        if state is None:
            zeros = torch.zeros(
                self.num_layers * self.num_directions,
                B, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )
            state = (zeros, zeros.clone())

        h_0, c_0 = state
        all_h: list = []
        all_c: list = []
        layer_input = x

        for layer in range(self.num_layers):
            fwd_idx = layer * self.num_directions
            h_fwd, c_fwd = h_0[fwd_idx], c_0[fwd_idx]
            fwd_cell = self.cells[fwd_idx]

            fwd_outputs = []
            for t in range(T):
                h_fwd, c_fwd = fwd_cell(layer_input[t], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)
            fwd_stack = torch.stack(fwd_outputs, dim=0)   # (T, B, H)

            if self.bidirectional:
                bwd_idx  = fwd_idx + 1
                h_bwd, c_bwd = h_0[bwd_idx], c_0[bwd_idx]
                bwd_cell = self.cells[bwd_idx]
                bwd_outputs = []
                for t in reversed(range(T)):
                    h_bwd, c_bwd = bwd_cell(layer_input[t], (h_bwd, c_bwd))
                    bwd_outputs.append(h_bwd)
                bwd_stack = torch.stack(bwd_outputs[::-1], dim=0)
                layer_out = torch.cat([fwd_stack, bwd_stack], dim=-1)
                all_h.extend([h_fwd, h_bwd])
                all_c.extend([c_fwd, c_bwd])
            else:
                layer_out = fwd_stack
                all_h.append(h_fwd)
                all_c.append(c_fwd)

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_out = self.dropout(layer_out)

            layer_input = layer_out

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)

        h_n = torch.stack(all_h, dim=0)
        c_n = torch.stack(all_c, dim=0)

        return output, (h_n, c_n)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, bidirectional={self.bidirectional}"
        )
