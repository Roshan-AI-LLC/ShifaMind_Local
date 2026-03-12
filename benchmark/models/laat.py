"""
benchmark/models/laat.py

LAAT — Label Attention for ICD Coding.
Reference: Vu et al. (2020) "Label-Attention Model for ICD Coding from
           Clinical Text." IJCAI 2020.

Architecture (faithful to Colab reference implementation):
  • Learned word embeddings (embed_dim=100) over BERT WordPiece vocab.
  • BiLSTM (hidden_dim=256 per direction → 512 total) over token sequence.
  • W_attn: linear projection of BiLSTM outputs → attention keys.
  • Per-label attention: label_queries [K, 512] attend over projected keys
    via einsum, producing label-specific document vectors [B, K, 512].
  • Per-label logit = dot(m_k, output_weight_k) + output_bias_k.
  • No chunking — tokenizer already truncates to max_length=512.
"""
import torch
import torch.nn as nn


class LAAT(nn.Module):
    """
    Label-Attention Model for multi-label ICD coding.

    Args:
        vocab_size    : tokeniser vocabulary size
        num_labels    : number of output codes (50)
        embed_dim     : word embedding dimension (default 100)
        hidden_dim    : BiLSTM hidden size PER DIRECTION (default 256 → 512 total)
        dropout       : dropout probability
        pad_token_id  : padding token id
    """

    def __init__(
        self,
        vocab_size:   int,
        num_labels:   int,
        embed_dim:    int   = 100,
        hidden_dim:   int   = 256,
        dropout:      float = 0.3,
        pad_token_id: int   = 0,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.d_model    = hidden_dim * 2   # BiLSTM concatenates both directions

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = 1,
            batch_first  = True,
            bidirectional= True,
        )

        # Attention projection: H → attention keys
        self.W_attn = nn.Linear(self.d_model, self.d_model, bias=False)

        # Per-label query vectors (learned)
        self.label_queries = nn.Parameter(torch.randn(num_labels, self.d_model))

        # Per-label classifier
        self.output_weight = nn.Parameter(torch.randn(num_labels, self.d_model))
        self.output_bias   = nn.Parameter(torch.zeros(num_labels))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_attn.weight)
        nn.init.normal_(self.label_queries, std=0.02)
        nn.init.normal_(self.output_weight, std=0.02)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,                               # absorbs chunk_size / chunk_overlap
    ) -> dict:
        """
        Args:
            input_ids      : [B, L]
            attention_mask : [B, L]  (unused — BiLSTM handles padding implicitly)
        Returns:
            dict with "logits" : [B, num_labels]
        """
        x = self.embedding(input_ids)           # [B, L, E]
        H, _ = self.lstm(x)                     # [B, L, 2H]

        # Projected keys for attention
        H_proj = self.W_attn(H)                 # [B, L, 2H]

        # Label-specific attention scores
        # einsum 'bth,lh->blt': [B, L, 2H] × [K, 2H] → [B, K, L]
        scores = torch.einsum('bth,lh->blt', H_proj, self.label_queries)
        alpha  = torch.softmax(scores, dim=2)   # [B, K, L]

        # Label-specific document vectors
        m = torch.bmm(alpha, H)                 # [B, K, 2H]

        # Per-label logits: element-wise dot + bias
        logits = (m * self.output_weight.unsqueeze(0)).sum(-1) + self.output_bias   # [B, K]

        return {"logits": logits}
