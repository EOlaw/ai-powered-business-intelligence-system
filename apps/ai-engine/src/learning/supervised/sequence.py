"""
InsightSerenity AI Engine — Sequence Labeler
=============================================
Assigns a label to every token in a sequence. The core architecture for:
    NER (Named Entity Recognition): classify each word as PER, ORG, LOC, O…
    POS tagging: classify each word as NOUN, VERB, ADJ…
    Chunking, slot filling, any other token classification task.

Architecture: BiLSTM → Linear(D → num_labels) with optional CRF layer.

Why BiLSTM for sequence labeling?
    Bidirectional LSTM reads the sequence in both directions, giving each
    token representation context from both left and right. "Washington" in
    "Washington DC" vs "George Washington" looks different with full context.

Why CRF (Conditional Random Field)?
    The CRF adds a transition score matrix learned during training.
    It enforces valid label sequences — e.g. in BIO tagging "I-PER" cannot
    follow "B-LOC". This prevents isolated invalid label predictions.

Without CRF: each token is classified independently → can produce B-LOC I-PER
With CRF:    the model learns B-LOC can't transition to I-PER → correct output

Training: NLL loss from CRF (or cross-entropy without CRF).
Inference: Viterbi algorithm finds the highest-scoring valid label sequence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.architectures.rnn.lstm import LSTM
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CRF layer
# ─────────────────────────────────────────────────────────────────────────────

class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field for sequence labeling.

    Adds a (num_labels × num_labels) transition score matrix on top of
    emission scores (LSTM output). During training, maximises the log
    likelihood of the correct label sequence using forward-backward algorithm.
    During inference, Viterbi finds the best label sequence.

    Args:
        num_labels: Total number of label classes (including START/STOP if used).
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels

        # Transition scores: transitions[i][j] = score of going from label i to label j
        # Initialised to small random values — the model learns the constraints
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.1)

        # Force padding index (0) to never be a valid start or destination
        # by initialising those paths to large negative values
        self.transitions.data[:, 0] = -10000.0   # Nothing transitions TO pad
        self.transitions.data[0, :] = -10000.0   # Pad doesn't transition FROM

    def forward(
        self,
        emissions: Tensor,   # (B, T, num_labels)
        labels:    Tensor,   # (B, T) — gold labels
        mask:      Tensor,   # (B, T) — 1 for real, 0 for pad
    ) -> Tensor:
        """Compute the negative log-likelihood (training loss)."""
        log_likelihood = self._score_sequence(emissions, labels, mask) \
                        - self._log_partition(emissions, mask)
        return -log_likelihood.mean()

    def decode(self, emissions: Tensor, mask: Tensor) -> List[List[int]]:
        """Viterbi decoding: find the best label sequence for each example."""
        B, T, _ = emissions.shape
        results  = []

        for b in range(B):
            seq_len = mask[b].sum().item()
            em      = emissions[b, :seq_len]   # (T, num_labels)
            result  = self._viterbi(em)
            results.append(result)

        return results

    def _score_sequence(self, emissions, labels, mask):
        """Compute the score of the gold label sequence."""
        B, T, _ = emissions.shape
        score   = emissions[:, 0].gather(1, labels[:, 0:1]).squeeze(1)

        for t in range(1, T):
            trans_score = self.transitions[labels[:, t-1], labels[:, t]]
            emit_score  = emissions[:, t].gather(1, labels[:, t:t+1]).squeeze(1)
            score       = score + (trans_score + emit_score) * mask[:, t].float()

        return score

    def _log_partition(self, emissions, mask):
        """Compute the log partition function Z using the forward algorithm."""
        B, T, L = emissions.shape
        # Initialise forward variables at t=0
        alpha = emissions[:, 0]   # (B, L)

        for t in range(1, T):
            # alpha: (B, L, 1) + transitions (L, L) + emissions (B, 1, L)
            scores  = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha_t = torch.logsumexp(scores, dim=1) + emissions[:, t]
            alpha   = torch.where(mask[:, t:t+1].bool(), alpha_t, alpha)

        return torch.logsumexp(alpha, dim=1)   # (B,)

    def _viterbi(self, emissions: Tensor) -> List[int]:
        """Viterbi algorithm for one sequence."""
        T, L     = emissions.shape
        viterbi  = emissions[0].clone()
        backptrs = []

        for t in range(1, T):
            scores = viterbi.unsqueeze(1) + self.transitions   # (L, L)
            best_scores, best_labels = scores.max(dim=0)
            backptrs.append(best_labels)
            viterbi = best_scores + emissions[t]

        # Traceback
        best_last = viterbi.argmax().item()
        path      = [best_last]
        for bp in reversed(backptrs):
            best_last = bp[best_last].item()
            path.append(best_last)

        return list(reversed(path))


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Labeler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SequenceLabelerConfig:
    vocab_size:   int
    num_labels:   int
    embed_dim:    int            = 64
    hidden_dim:   int            = 128
    num_layers:   int            = 2
    dropout:      float          = 0.3
    use_crf:      bool           = True
    lr:           float          = 1e-3
    batch_size:   int            = 32
    epochs:       int            = 30
    device:       str            = "cpu"


class SequenceLabeler(nn.Module):
    """
    BiLSTM + CRF sequence labeler for NER, POS, and other token tasks.

    Input:  (B, T) integer token IDs
    Output: (B, T) predicted label indices

    Args:
        config: SequenceLabelerConfig.
    """

    def __init__(self, config: SequenceLabelerConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.bilstm    = LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim // 2,   # BiLSTM doubles: //2 per direction
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout   = nn.Dropout(p=config.dropout)
        # Emission scores: hidden → num_labels logits per token
        self.fc        = nn.Linear(config.hidden_dim, config.num_labels)
        self.crf       = CRF(config.num_labels) if config.use_crf else None
        self._is_fitted = False

    def forward(self, input_ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute emission scores for each token.

        Args:
            input_ids: (B, T) token IDs.
            mask:      (B, T) — 1 for real, 0 for padding.

        Returns:
            (B, T, num_labels) emission scores.
        """
        emb    = self.embedding(input_ids)        # (B, T, E)
        emb    = self.dropout(emb)
        out, _ = self.bilstm(emb)                 # (B, T, H)
        out    = self.dropout(out)
        return self.fc(out)                       # (B, T, L)

    def fit(
        self,
        X: Union[np.ndarray, Tensor],    # (N, T) token IDs
        y: Union[np.ndarray, Tensor],    # (N, T) label IDs
        mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> "SequenceLabeler":
        """
        Train the sequence labeler.

        Args:
            X:    Integer token ID sequences. Shape (N, T).
            y:    Integer label sequences. Shape (N, T).
            mask: Boolean mask. Shape (N, T). If None, inferred from X != 0.
        """
        device = torch.device(self.config.device)
        self.to(device)

        X_t = self._to_long(X).to(device)
        y_t = self._to_long(y).to(device)

        if mask is None:
            mask_t = (X_t != 0).long()
        else:
            mask_t = self._to_long(mask).to(device)

        loader = DataLoader(
            TensorDataset(X_t, y_t, mask_t),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            self.train()
            ep_loss = 0.0

            for X_b, y_b, m_b in loader:
                emissions = self.forward(X_b, m_b)

                if self.crf is not None:
                    loss = self.crf(emissions, y_b, m_b)
                else:
                    # Flatten for CrossEntropy
                    B, T, L = emissions.shape
                    flat_em  = emissions.reshape(B * T, L)
                    flat_y   = y_b.reshape(B * T)
                    loss     = nn.CrossEntropyLoss(ignore_index=0)(flat_em, flat_y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                logger.info("Sequence labeler training",
                            epoch=epoch + 1,
                            loss=round(ep_loss / len(loader), 4))

        self._is_fitted = True
        return self

    def predict(
        self,
        X: Union[np.ndarray, Tensor],
        mask: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> List[List[int]]:
        """
        Predict label sequences for each input.

        Returns:
            List of label ID lists (one per sequence, padded positions omitted).
        """
        device = torch.device(self.config.device)
        X_t    = self._to_long(X).to(device)
        mask_t = ((X_t != 0).long() if mask is None else self._to_long(mask).to(device))

        self.eval()
        with torch.no_grad():
            emissions = self.forward(X_t, mask_t)

        if self.crf is not None:
            return self.crf.decode(emissions, mask_t)

        # Greedy decode (no CRF)
        preds = emissions.argmax(dim=-1)   # (B, T)
        result = []
        for b in range(preds.shape[0]):
            seq_len = mask_t[b].sum().item()
            result.append(preds[b, :seq_len].tolist())
        return result

    @staticmethod
    def _to_long(arr) -> Tensor:
        if isinstance(arr, Tensor):
            return arr.long()
        return torch.tensor(arr, dtype=torch.long)
