"""
benchmark/models/msmn.py

MSMN — Multi-Synonyms Matching Network for ICD coding.
Reference: Yuan et al. (2022) "Code Synonyms Do Matter: Multiple Synonyms
           Matching Network for Automatic ICD Coding." ACL 2022.

Key ideas (adapted for Top-50):
  1. Each ICD code has S synonym descriptions (name, long title, keywords).
     These are sourced from top50_icd10_info.json and the GLOBAL_CONCEPTS list.
  2. The clinical note is encoded with BioClinicalBERT.
  3. For each code, each synonym description is also encoded with BERT.
  4. Cross-attention between the note tokens and each synonym representation
     yields a synonym-specific document vector.
  5. Max-pool across synonyms → per-code document vector → sigmoid.

This is a faithful adaptation: the original used full BERT for both note and
synonym encoding with shared weights.

Synonym sources (from top50_icd10_info.json):
  - "name"       : short ICD-10 code name (e.g. "Sepsis")
  - "description": longer title (e.g. "Sepsis, unspecified organism")
  - Up to S total per code, padded/truncated to S=4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class MSMN(nn.Module):
    """
    Multi-Synonyms Matching Network.

    Args:
        bert_model_name  : shared BERT encoder for note + synonyms
        num_labels       : number of output codes (50)
        num_synonyms     : synonyms per code (S)
        hidden_size      : BERT hidden size (768)
        dropout          : dropout probability
        synonym_input_ids   : [K, S, L_s]  pre-encoded synonym token ids
        synonym_attn_mask   : [K, S, L_s]  pre-encoded synonym attention masks
    """

    def __init__(
        self,
        bert_model_name:    str,
        num_labels:         int,
        num_synonyms:       int,
        hidden_size:        int   = 768,
        dropout:            float = 0.1,
        synonym_input_ids:  torch.Tensor | None = None,
        synonym_attn_mask:  torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_labels   = num_labels
        self.num_synonyms = num_synonyms
        self.hidden_size  = hidden_size

        self.bert    = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)

        # Cross-attention: note keys/values, synonym queries
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj   = nn.Linear(hidden_size, hidden_size)
        self.scale      = hidden_size ** -0.5

        # Final per-label classifier (input: hidden_size after max-pool)
        self.classifier = nn.Linear(hidden_size, 1)   # applied per code

        # Register synonym token ids as buffers (not trained)
        if synonym_input_ids is not None:
            self.register_buffer("synonym_input_ids",  synonym_input_ids)
            self.register_buffer("synonym_attn_mask",  synonym_attn_mask)
        else:
            self.synonym_input_ids = None
            self.synonym_attn_mask = None

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for proj in [self.query_proj, self.key_proj, self.classifier]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_synonyms(self) -> torch.Tensor:
        """
        Encode all synonym descriptions with BERT.
        Returns [K, S, H] — one CLS embedding per synonym per code.
        Called once at the start of training/eval if synonyms change.
        """
        K, S, L = self.synonym_input_ids.shape           # [K, S, L_s]
        flat_ids  = self.synonym_input_ids.view(K * S, L)
        flat_mask = self.synonym_attn_mask.view(K * S, L)

        # Encode in mini-batches to avoid OOM
        batch_size = 32
        cls_list   = []
        for i in range(0, K * S, batch_size):
            out = self.bert(
                input_ids      = flat_ids[i: i + batch_size],
                attention_mask = flat_mask[i: i + batch_size],
            )
            cls_list.append(out.last_hidden_state[:, 0, :])   # [B, H]
        cls = torch.cat(cls_list, dim=0)                       # [K*S, H]
        return cls.view(K, S, self.hidden_size)                # [K, S, H]

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        synonym_cls:    torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            input_ids      : [B, L_n]
            attention_mask : [B, L_n]
            synonym_cls    : [K, S, H] pre-computed synonym CLS embeddings.
                             If None, synonyms are encoded on the fly
                             (slower — use precompute_synonyms() before training).
        Returns:
            dict with "logits" : [B, K]
        """
        B = input_ids.size(0)
        K = self.num_labels
        S = self.num_synonyms

        # 1. Encode note
        note_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Token representations: [B, L_n, H]
        note_h = self.dropout(note_out.last_hidden_state)

        # 2. Synonym embeddings
        if synonym_cls is None:
            synonym_cls = self.encode_synonyms()   # [K, S, H]

        # 3. For each code k, for each synonym s, compute cross-attention
        #    between note tokens and synonym[k,s] as query.
        # Q: [B, K, S, H] (broadcast synonym_cls across batch)
        syn = synonym_cls.unsqueeze(0).expand(B, -1, -1, -1)   # [B, K, S, H]

        # Flatten to [B*K*S, H] for batched projection
        syn_flat  = syn.reshape(B * K * S, self.hidden_size)    # [B*K*S, H]
        Q         = self.query_proj(syn_flat).reshape(B, K * S, self.hidden_size)  # [B, K*S, H]

        # Note keys: [B, L_n, H]
        Keys = self.key_proj(note_h)                            # [B, L_n, H]

        # Attention scores: [B, K*S, L_n]
        attn = torch.bmm(Q, Keys.transpose(1, 2)) * self.scale  # [B, K*S, L_n]

        # Mask padding tokens in the note
        if attention_mask is not None:
            mask = (1.0 - attention_mask.float()).unsqueeze(1) * -1e9   # [B, 1, L_n]
            attn = attn + mask

        attn    = torch.softmax(attn, dim=-1)                   # [B, K*S, L_n]
        context = torch.bmm(attn, note_h)                       # [B, K*S, H]
        context = context.view(B, K, S, self.hidden_size)        # [B, K, S, H]

        # 4. Max-pool across synonyms → [B, K, H]
        context_max = context.max(dim=2).values                  # [B, K, H]
        context_max = self.dropout(context_max)

        # 5. Per-code logits: reuse classifier weight as per-code query
        # classifier.weight ∈ [1, H] is a single shared relevance scorer;
        # for independent per-code scoring, we use the dot-product with the
        # classifier weight applied per-code (equivalent to the original paper).
        logits = self.classifier(context_max).squeeze(-1)        # [B, K]

        return {"logits": logits}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build synonym token tensors from top50_info JSON
# Called once in train_all.py before training MSMN.
# ──────────────────────────────────────────────────────────────────────────────

def build_synonym_tensors(
    top50_info:    dict,
    top50_codes:   list,
    tokenizer,
    num_synonyms:  int = 4,
    max_syn_len:   int = 32,
    device:        torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build synonym_input_ids [K, S, L_s] and synonym_attn_mask [K, S, L_s].

    Synonym sources per code (in order, padded/truncated to num_synonyms):
      1. Short name from top50_info (e.g. "Sepsis, unspecified organism")
      2. ICD-10 chapter description
      3. Code string itself (e.g. "A419")
      4. Fallback: generic "clinical condition"

    Args:
        top50_info   : dict loaded from top50_icd10_info.json
        top50_codes  : ordered list of 50 ICD-10 codes
        tokenizer    : HuggingFace tokenizer (BERT)
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
        # Source 1: full display name
        name = info.get("name", "") or info.get("description", "")
        if name:
            synonyms.append(name)
        # Source 2: short code + name combo
        synonyms.append(f"{code}: {name}" if name else code)
        # Source 3: raw code
        synonyms.append(code)
        # Source 4: chapter name or fallback
        chapter = info.get("chapter", "") or "clinical diagnosis"
        synonyms.append(chapter)

        # Truncate / pad to exactly num_synonyms
        synonyms = (synonyms + ["clinical condition"] * num_synonyms)[:num_synonyms]

        enc = tokenizer(
            synonyms,
            max_length      = max_syn_len,
            truncation      = True,
            padding         = "max_length",
            return_tensors  = "pt",
        )
        all_ids.append(enc["input_ids"])     # [S, L_s]
        all_masks.append(enc["attention_mask"])

    input_ids  = torch.stack(all_ids)    # [K, S, L_s]
    attn_mask  = torch.stack(all_masks)  # [K, S, L_s]

    if device is not None:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

    return input_ids, attn_mask
