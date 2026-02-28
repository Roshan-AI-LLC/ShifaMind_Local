"""
data/datasets.py

Two dataset classes used across all phases:

  ConceptDataset  — Phase 1 & 2  (returns dx labels + concept labels)
  RAGDataset      — Phase 3       (also returns raw text for RAG retrieval)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import config


class ConceptDataset(Dataset):
    """
    Tokenises clinical notes and returns:
        input_ids       [seq_len]
        attention_mask  [seq_len]
        labels          [num_dx]        float  (Top-50 ICD-10 multi-hot)
        concept_labels  [num_concepts]  float  (111 keyword concept multi-hot)
    """

    def __init__(self, texts, labels, concept_labels, tokenizer, max_length: int = None):
        self.texts          = texts
        self.labels         = labels
        self.concept_labels = concept_labels
        self.tokenizer      = tokenizer
        self.max_length     = max_length or config.MAX_LENGTH

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.FloatTensor(self.labels[idx]),
            "concept_labels": torch.FloatTensor(self.concept_labels[idx]),
        }


class RAGDataset(Dataset):
    """
    Phase 3 dataset.

    Same as ConceptDataset but also returns the raw text string so the
    RAG retriever can look up relevant passages at training/inference time.

    Expects *df* to have ICD-10 code columns (e.g. 'I50', 'J18', …) that
    are listed in *top50_codes*.
    """

    def __init__(self, df, tokenizer, concept_labels, top50_codes, max_length: int = None):
        self.texts          = df["text"].tolist()
        self.labels         = df[top50_codes].values.astype(np.float32)  # [N, 50]
        self.concept_labels = concept_labels                              # [N, 111]
        self.tokenizer      = tokenizer
        self.max_length     = max_length or config.MAX_LENGTH

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "text"          : text,
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.float),
            "concept_labels": torch.tensor(self.concept_labels[idx], dtype=torch.float),
        }
