"""
benchmark/models/caml.py

CAML — Convolutional Attention Model for ICD coding.
Reference: Mullenbach et al. (2018) "Explainable Prediction of Medical Codes
           from Clinical Text." NAACL 2018.

Faithful to the paper:
  • Learned word embeddings (embed_dim=100) over BERT WordPiece vocab.
    Using BERT's tokenizer ensures identical text preprocessing to ShifaMind.
  • 1D convolution (filter_size=4, num_filters=500) over the token sequence.
  • Per-label attention: for each label i, a learned vector u_i attends over
    the convolutional hidden states to produce a label-specific document
    representation, then a linear layer maps it to a logit.
  • Binary cross-entropy loss (no focal loss — faithful to original).
  • No chunking — input is truncated to max_length=512 by the tokenizer,
    matching the original paper's fixed-length approach.
"""
import torch
import torch.nn as nn


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

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )

        # padding='same' keeps output length == input length for any kernel size
        self.conv = nn.Conv1d(
            in_channels  = embed_dim,
            out_channels = num_filters,
            kernel_size  = filter_size,
            padding      = "same",
        )

        # Per-label attention weights  U ∈ R^{num_labels × num_filters}
        self.label_attn = nn.Linear(num_filters, num_labels, bias=False)

        # Per-label classification  W ∈ R^{num_labels × num_filters}
        self.output     = nn.Linear(num_filters, num_labels)

        self.dropout    = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.label_attn.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            input_ids      : [B, L]  — already truncated to 512 by tokenizer
            attention_mask : [B, L]  — unused, kept for API compatibility
        Returns:
            dict with key "logits" : [B, num_labels]
        """
        x = self.dropout(self.embedding(input_ids))   # [B, L, E]
        h = torch.tanh(self.conv(x.transpose(1, 2)))  # [B, F, L]
        h = h.transpose(1, 2)                         # [B, L, F]

        alpha = torch.softmax(self.label_attn(h), dim=1)         # [B, L, K]
        v     = self.dropout(torch.bmm(alpha.transpose(1, 2), h)) # [B, K, F]
        logits = (v * self.output.weight.unsqueeze(0)).sum(-1) + self.output.bias

        return {"logits": logits}
