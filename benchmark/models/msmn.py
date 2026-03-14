"""
benchmark/models/msmn.py

MSMN — Multi-Synonyms Matching Network for ICD coding.
Reference: Yuan et al. (2022) "Code Synonyms Do Matter: Multiple Synonyms
           Matching Network for Automatic ICD Coding." ACL 2022.
           https://github.com/GanjinZero/ICD-MSMN

Architecture (faithful to Yuan et al. 2022):
  • Shared TextEncoder: learned word embeddings (embed_dim=100) +
    BiLSTM (hidden_dim=512 per direction, 1024 total).
    The SAME encoder is used for both the clinical note and synonym text.
  • LabelEncoder: each synonym description is encoded by TextEncoder,
    then max-pooled over sequence positions → per-synonym feature [2H].
    Synonyms per code are max-pooled → per-label feature [2H].
    Projected to attention_dim via label_proj → [attention_dim].
  • MultiLabelMultiHeadLAAT (MultiHeadLAAT with synonym-derived label queries):
      z     = tanh(W_attn · h)                       [B, L, att_dim_head]
      u     = label_feat.view(K, n_head, att_dim_head)
      score = einsum('bld,knd->bkln', z, u)          [B, K, L, n_head]
      alpha = softmax(score, dim=2)                  [B, K, L, n_head]
      m     = einsum('bld,bkln->bkdn', h, alpha)     [B, K, 2H, n_head]
      m     = max(m, dim=-1)                         [B, K, 2H]
  • Per-label element-wise classifier: final.weight ⊙ m, sum over H.
"""
import torch
import torch.nn as nn


class MSMN(nn.Module):
    """
    Multi-Synonyms Matching Network.

    Args:
        vocab_size        : tokeniser vocabulary size
        num_labels        : number of output codes (50)
        num_synonyms      : synonyms per code (S)
        embed_dim         : word embedding dimension (100)
        hidden_dim        : BiLSTM hidden size PER DIRECTION (512; total = 1024)
        attention_dim     : total multi-head attention dimension (512)
        attention_head    : number of attention heads (4)
        dropout           : representation dropout (rep_dropout = 0.2)
        lstm_dropout      : inter-layer BiLSTM dropout (unused for 1 layer)
        pad_token_id      : padding token id for embedding
        synonym_input_ids : [K, S, L_s]  synonym token ids (registered buffer)
        synonym_attn_mask : [K, S, L_s]  synonym attention masks (registered buffer)
    """

    def __init__(
        self,
        vocab_size:         int,
        num_labels:         int,
        num_synonyms:       int,
        embed_dim:          int   = 100,
        hidden_dim:         int   = 512,
        attention_dim:      int   = 512,
        attention_head:     int   = 4,
        dropout:            float = 0.2,
        lstm_dropout:       float = 0.1,
        pad_token_id:       int   = 0,
        synonym_input_ids:  torch.Tensor | None = None,
        synonym_attn_mask:  torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert attention_dim % attention_head == 0, \
            "attention_dim must be divisible by attention_head"

        self.num_labels     = num_labels
        self.num_synonyms   = num_synonyms
        self.hidden_dim     = hidden_dim
        self.attention_head = attention_head
        self.att_dim_head   = attention_dim // attention_head   # per-head dim

        # ── Shared text encoder (document + synonyms) ────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = True,
        )
        self.dropout = nn.Dropout(dropout)

        # ── LabelEncoder: project synonym BiLSTM output → attention_dim ─────
        self.label_proj = nn.Linear(hidden_dim * 2, attention_dim, bias=False)

        # ── MultiHeadLAAT attention ──────────────────────────────────────────
        # W_attn projects document tokens to per-head query space
        self.W_attn = nn.Linear(hidden_dim * 2, self.att_dim_head, bias=False)

        # ── Per-label element-wise classifier ────────────────────────────────
        self.final = nn.Linear(hidden_dim * 2, num_labels)

        # Synonym token ids (not trained — registered as buffers)
        if synonym_input_ids is not None:
            self.register_buffer("synonym_input_ids", synonym_input_ids)
            self.register_buffer("synonym_attn_mask", synonym_attn_mask)
        else:
            self.synonym_input_ids = None
            self.synonym_attn_mask = None

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.label_proj.weight)
        nn.init.xavier_uniform_(self.W_attn.weight)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    # ------------------------------------------------------------------
    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shared text encoder: word embedding + BiLSTM + dropout.
        Args:
            input_ids : [B, L]
        Returns:
            hidden    : [B, L, 2 * hidden_dim]
        """
        x      = self.dropout(self.embedding(input_ids))   # [B, L, E]
        h, _   = self.lstm(x)                              # [B, L, 2H]
        return self.dropout(h)

    # ------------------------------------------------------------------
    def _label_features(self) -> torch.Tensor:
        """
        Encode synonym descriptions → per-label feature vectors.

        Steps (Yuan et al. 2022, LabelEncoder):
          1. Encode each synonym via shared TextEncoder → [K*S, L_s, 2H]
          2. Max-pool over sequence positions → per-synonym feature [K*S, 2H]
          3. Reshape to [K, S, 2H], max-pool over synonyms → [K, 2H]
          4. Project to attention_dim → [K, attention_dim]

        Returns:
            label_feat : [K, attention_dim]
        """
        K, S, L = self.synonym_input_ids.shape
        flat    = self.synonym_input_ids.view(K * S, L)    # [K*S, L_s]

        h    = self._encode(flat)                          # [K*S, L_s, 2H]
        feat = h.max(dim=1).values                         # [K*S, 2H]  max over seq
        feat = feat.view(K, S, self.hidden_dim * 2)        # [K, S, 2H]
        feat = feat.max(dim=1).values                      # [K, 2H]    max over synonyms
        return self.label_proj(feat)                       # [K, attention_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Args:
            input_ids      : [B, L]
            attention_mask : [B, L]
        Returns:
            dict with "logits" : [B, num_labels]
        """
        B  = input_ids.size(0)
        K  = self.num_labels
        nH = self.attention_head
        dH = self.att_dim_head

        # 1. Encode document
        h = self._encode(input_ids)                        # [B, L, 2H]

        # 2. Encode synonyms → per-label features
        label_feat = self._label_features()                # [K, attention_dim]

        # 3. MultiLabelMultiHeadLAAT attention
        # Project document tokens to per-head attention space
        z = torch.tanh(self.W_attn(h))                     # [B, L, att_dim_head]

        # Reshape label features to multi-head: [K, n_head, att_dim_head]
        u = label_feat.view(K, nH, dH)

        # Attention scores: einsum 'bld,knd->bkln' → [B, K, L, n_head]
        score = torch.einsum('bld,knd->bkln', z, u)

        # Mask padding positions
        if attention_mask is not None:
            pad_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            score    = score.masked_fill(pad_mask, -1e9)

        alpha = torch.softmax(score, dim=2)                # [B, K, L, n_head]

        # Weighted sum: einsum 'bld,bkln->bkdn' → [B, K, 2H, n_head]
        m = torch.einsum('bld,bkln->bkdn', h, alpha)

        # Max-pool over heads → [B, K, 2H]
        m = m.max(dim=-1).values
        m = self.dropout(m)

        # 4. Per-label element-wise classification (same pattern as CAML/LAAT)
        logits = (m * self.final.weight.unsqueeze(0)).sum(-1) + self.final.bias  # [B, K]

        return {"logits": logits}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build synonym token tensors from top50_info JSON
# Called once in train_all.py / evaluate_all.py before building MSMN.
# ──────────────────────────────────────────────────────────────────────────────

def build_synonym_tensors(
    top50_info:    dict,
    top50_codes:   list,
    tokenizer,
    num_synonyms:  int = 8,
    max_syn_len:   int = 32,
    device:        torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build synonym_input_ids [K, S, L_s] and synonym_attn_mask [K, S, L_s].

    Synonym sources per code (in order, padded/truncated to num_synonyms):
      1. Short name from top50_info (e.g. "Sepsis, unspecified organism")
      2. ICD-10 chapter description
      3. Code string itself (e.g. "A419")
      4. Fallback: "clinical condition"

    Args:
        top50_info   : dict loaded from top50_icd10_info.json
        top50_codes  : ordered list of 50 ICD-10 codes
        tokenizer    : HuggingFace tokenizer (shared BERT WordPiece)
        num_synonyms : S — synonyms per code
        max_syn_len  : max token length per synonym
        device       : target device

    Returns:
        (input_ids, attn_mask) each [K, S, L_s]
    """
    all_ids, all_masks = [], []

    for code in top50_codes:
        info = top50_info.get(code, {})

        synonyms = []
        name     = info.get("name", "") or info.get("description", "")
        if name:
            synonyms.append(name)
        synonyms.append(f"{code}: {name}" if name else code)
        synonyms.append(code)
        chapter = info.get("chapter", "") or "clinical diagnosis"
        synonyms.append(chapter)

        # Pad/truncate to exactly num_synonyms
        synonyms = (synonyms + ["clinical condition"] * num_synonyms)[:num_synonyms]

        enc = tokenizer(
            synonyms,
            max_length     = max_syn_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt",
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])

    input_ids = torch.stack(all_ids)    # [K, S, L_s]
    attn_mask = torch.stack(all_masks)  # [K, S, L_s]

    if device is not None:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

    return input_ids, attn_mask
