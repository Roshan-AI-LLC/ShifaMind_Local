"""
benchmark/models/caml.py

CAML — Convolutional Attention Model for ICD coding.
Reference: Mullenbach et al. (2018) "Explainable Prediction of Medical Codes
           from Clinical Text." NAACL 2018.

Key design choices faithful to the paper:
  • Learned word embeddings (embed_dim=100) over BERT WordPiece vocab.
    Using BERT's tokenizer ensures identical text preprocessing to ShifaMind.
  • 1D convolution (filter_size=4, num_filters=500) over the token sequence.
  • Per-label attention: for each label i, a learned vector u_i attends over
    the convolutional hidden states to produce a label-specific document
    representation, then a linear layer maps it to a logit.
  • Binary cross-entropy loss (no focal loss — faithful to original).

Long-document handling:
  • Documents are chunked into overlapping windows of chunk_size tokens.
  • For each label, attention and logits are computed per chunk.
  • Final logit = max over chunks (captures the most relevant section).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAML(nn.Module):
    """
    Convolutional Attention Multi-Label model for Top-N ICD coding.

    Args:
        vocab_size    : size of the tokeniser vocabulary
        num_labels    : number of output codes (50)
        embed_dim     : word embedding dimension (default 100)
        num_filters   : number of CNN filters (default 500)
        filter_size   : convolution kernel width (default 4)
        dropout       : dropout probability
        pad_token_id  : padding token id (used to initialise embedding to 0)
    """

    def __init__(
        self,
        vocab_size:   int,
        num_labels:   int,
        embed_dim:    int = 100,
        num_filters:  int = 500,
        filter_size:  int = 4,
        dropout:      float = 0.3,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_labels  = num_labels
        self.num_filters = num_filters

        # Word embeddings — Xavier uniform init (default for nn.Embedding)
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )

        # 1-D convolution — padding='same' asks PyTorch to compute the correct
        # (possibly asymmetric) padding so that output length == input length for
        # any kernel size, including even sizes like 4.
        # The old padding=filter_size//2 gave L+1 output for filter_size=4
        # (an extra ghost token the attention could attend to).
        self.conv = nn.Conv1d(
            in_channels  = embed_dim,
            out_channels = num_filters,
            kernel_size  = filter_size,
            padding      = "same",
        )

        # Per-label attention weights  U ∈ R^{num_labels × num_filters}
        # Each row u_i is a label-specific query vector
        self.label_attn = nn.Linear(num_filters, num_labels, bias=False)

        # Per-label classification  W ∈ R^{num_labels × num_filters}
        self.output     = nn.Linear(num_filters, num_labels)

        self.dropout    = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.label_attn.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # ------------------------------------------------------------------
    def _encode_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of token sequences and return per-label logits.

        Args:
            input_ids : [B, L]
        Returns:
            logits    : [B, num_labels]
        """
        # Embedding: [B, L, E]
        x = self.dropout(self.embedding(input_ids))

        # Conv expects [B, C, L]
        x = x.transpose(1, 2)                             # [B, E, L]
        h = torch.tanh(self.conv(x))                      # [B, F, L]
        h = h.transpose(1, 2)                             # [B, L, F]

        # Per-label attention scores: [B, L, num_labels]
        alpha = self.label_attn(h)                        # [B, L, K]
        alpha = torch.softmax(alpha, dim=1)               # [B, L, K]

        # Label-specific document vectors: [B, K, F]
        # v_i = sum_t alpha_{ti} * h_t
        v = torch.bmm(alpha.transpose(1, 2), h)           # [B, K, F]
        v = self.dropout(v)

        # Per-label logits: [B, K]
        # Re-use output.weight as the label classifier
        # output.weight is [K, F], so we do element-wise and sum over F
        logits = (v * self.output.weight.unsqueeze(0)).sum(-1) + self.output.bias
        return logits

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size:     int = 512,
        chunk_overlap:  int = 64,
    ) -> dict:
        """
        Forward pass with optional chunking for long documents.

        Args:
            input_ids      : [B, L]
            attention_mask : [B, L] (optional, used to strip padding)
            chunk_size     : max tokens per chunk
            chunk_overlap  : overlap between consecutive chunks
        Returns:
            dict with key "logits" : [B, num_labels]
        """
        B, L = input_ids.shape

        if L <= chunk_size:
            logits = self._encode_chunk(input_ids)
        else:
            # Chunk and max-pool across chunks
            stride   = chunk_size - chunk_overlap
            starts   = list(range(0, L - chunk_size + 1, stride))
            if not starts or starts[-1] + chunk_size < L:
                starts.append(max(0, L - chunk_size))

            chunk_logits = []
            for s in starts:
                chunk = input_ids[:, s: s + chunk_size]
                chunk_logits.append(self._encode_chunk(chunk))   # [B, K]

            # Max over chunks — captures the most relevant window per label
            logits = torch.stack(chunk_logits, dim=0).max(dim=0).values

        return {"logits": logits}
