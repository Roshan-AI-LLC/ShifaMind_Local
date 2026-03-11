"""
benchmark/models/laat.py

LAAT — Label Attention for ICD Coding.
Reference: Vu et al. (2020) "Label-Attention Model for ICD Coding from
           Clinical Text." IJCAI 2020.

Key design choices:
  • BiGRU over word embeddings (embed_dim=100) produces contextual token
    representations.
  • Per-label attention: learned label embeddings Q ∈ R^{K×d_q} attend
    over BiGRU outputs H ∈ R^{T×2H} via scaled dot-product attention.
  • Label-specific document vectors fed to per-label linear classifiers.
  • Same long-document chunking strategy as CAML (max-pool over chunks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LAAT(nn.Module):
    """
    Label-Attention Model for multi-label ICD coding.

    Args:
        vocab_size      : tokeniser vocabulary size
        num_labels      : number of output codes (50)
        embed_dim       : word embedding dimension (default 100)
        hidden_dim      : BiGRU hidden size per direction (default 512)
        label_embed_dim : label query embedding dimension (default 256)
        dropout         : dropout probability
        pad_token_id    : padding token id
    """

    def __init__(
        self,
        vocab_size:      int,
        num_labels:      int,
        embed_dim:       int   = 100,
        hidden_dim:      int   = 512,
        label_embed_dim: int   = 256,
        dropout:         float = 0.3,
        pad_token_id:    int   = 0,
    ) -> None:
        super().__init__()
        self.num_labels  = num_labels
        self.hidden_dim  = hidden_dim
        self.d_model     = hidden_dim * 2   # BiGRU concatenates both directions

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_token_id
        )
        self.bigru = nn.GRU(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = 1,
            batch_first  = True,
            bidirectional= True,
        )

        # Project BiGRU output to label_embed_dim for attention keys/values
        self.key_proj = nn.Linear(self.d_model, label_embed_dim)

        # Label query embeddings  Q ∈ R^{num_labels × label_embed_dim}
        self.label_queries = nn.Embedding(num_labels, label_embed_dim)
        self.scale = label_embed_dim ** -0.5

        # Per-label classification head
        self.output = nn.Linear(label_embed_dim, num_labels)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.label_queries.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        for name, param in self.bigru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    def _encode_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode one chunk and return per-label logits.

        Args:
            input_ids : [B, L]
        Returns:
            logits    : [B, num_labels]
        """
        B = input_ids.size(0)

        emb = self.dropout(self.embedding(input_ids))    # [B, L, E]
        h, _ = self.bigru(emb)                           # [B, L, 2H]
        h = self.dropout(h)

        # Project to key space: [B, L, D]
        keys = torch.tanh(self.key_proj(h))              # [B, L, D]

        # Label queries: [K, D] → [1, K, D] → [B, K, D]
        label_ids = torch.arange(self.num_labels, device=input_ids.device)
        Q = self.label_queries(label_ids)                # [K, D]
        Q = Q.unsqueeze(0).expand(B, -1, -1)             # [B, K, D]

        # Scaled dot-product attention: [B, K, L]
        attn = torch.bmm(Q, keys.transpose(1, 2)) * self.scale   # [B, K, L]
        attn = torch.softmax(attn, dim=-1)                        # [B, K, L]

        # Label-specific document vectors: [B, K, D]
        context = torch.bmm(attn, keys)                  # [B, K, D]
        context = self.dropout(context)

        # Diagonal logit extraction: per-label classification
        # output.weight [K, D]; we use each row for its label
        logits = (context * self.output.weight.unsqueeze(0)).sum(-1) + self.output.bias
        return logits                                     # [B, K]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size:     int = 512,
        chunk_overlap:  int = 64,
    ) -> dict:
        """
        Args:
            input_ids      : [B, L]
            attention_mask : [B, L]  (optional)
            chunk_size     : max tokens per chunk
            chunk_overlap  : overlap between consecutive chunks
        Returns:
            dict with "logits" : [B, num_labels]
        """
        B, L = input_ids.shape

        if L <= chunk_size:
            logits = self._encode_chunk(input_ids)
        else:
            stride = chunk_size - chunk_overlap
            starts = list(range(0, L - chunk_size + 1, stride))
            if not starts or starts[-1] + chunk_size < L:
                starts.append(max(0, L - chunk_size))

            chunk_logits = [
                self._encode_chunk(input_ids[:, s: s + chunk_size])
                for s in starts
            ]
            logits = torch.stack(chunk_logits, dim=0).max(dim=0).values

        return {"logits": logits}
