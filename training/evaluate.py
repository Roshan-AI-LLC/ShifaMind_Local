"""
training/evaluate.py

Evaluation helpers for each phase.

All functions:
  • set model to eval mode
  • run inference with torch.no_grad()
  • return a metrics dict ready for logging / saving
"""
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from training.loss import MultiObjectiveLoss


# ============================================================================
# PHASE 1
# ============================================================================

def evaluate_phase1(
    model,
    dataloader,
    criterion: MultiObjectiveLoss,
    device: torch.device,
) -> dict:
    """
    Evaluate Phase 1 (ShifaMind2Phase1) on a given dataloader.

    Returns:
        dict with keys: loss, dx_f1, concept_f1, loss_dx, loss_align, loss_concept
    """
    model.eval()
    dx_probs_list, dx_labels_list       = [], []
    concept_probs_list, concept_labels_list = [], []
    total_loss = 0.0
    loss_acc   = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dx_labels      = batch["labels"].to(device)
            concept_labels = batch["concept_labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss, comp = criterion(outputs, dx_labels, concept_labels)

            total_loss += loss.item()
            for k, v in comp.items():
                loss_acc[k] += v

            dx_probs_list.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            dx_labels_list.append(dx_labels.cpu().numpy())
            concept_probs_list.append(outputs["concept_scores"].cpu().numpy())
            concept_labels_list.append(concept_labels.cpu().numpy())

    n = len(dataloader)
    dx_preds    = np.vstack(dx_probs_list)
    dx_labels   = np.vstack(dx_labels_list)
    c_preds     = np.vstack(concept_probs_list)
    c_labels    = np.vstack(concept_labels_list)

    dx_f1      = f1_score(dx_labels,  (dx_preds  > 0.5).astype(int), average="macro", zero_division=0)
    concept_f1 = f1_score(c_labels,   (c_preds   > 0.5).astype(int), average="macro", zero_division=0)

    return {
        "loss"        : total_loss / n,
        "dx_f1"       : float(dx_f1),
        "concept_f1"  : float(concept_f1),
        "loss_dx"     : loss_acc["dx"]      / n,
        "loss_align"  : loss_acc["align"]   / n,
        "loss_concept": loss_acc["concept"] / n,
    }


# ============================================================================
# PHASE 2
# ============================================================================

def evaluate_phase2(
    model,
    dataloader,
    criterion: MultiObjectiveLoss,
    device: torch.device,
    concept_embeddings: torch.Tensor,
) -> dict:
    """
    Evaluate Phase 2 (ShifaMindPhase2GAT).

    concept_embeddings: [num_concepts, 768] detached tensor on the correct device.
    """
    model.eval()
    dx_probs_list, dx_labels_list           = [], []
    concept_probs_list, concept_labels_list = [], []
    total_loss = 0.0
    loss_acc   = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dx_labels      = batch["labels"].to(device)
            concept_labels = batch["concept_labels"].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)
            loss, comp = criterion(outputs, dx_labels, concept_labels)

            total_loss += loss.item()
            for k, v in comp.items():
                loss_acc[k] += v

            dx_probs_list.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            dx_labels_list.append(dx_labels.cpu().numpy())
            concept_probs_list.append(outputs["concept_scores"].cpu().numpy())
            concept_labels_list.append(concept_labels.cpu().numpy())

    n = len(dataloader)
    dx_preds  = np.vstack(dx_probs_list)
    dx_labels = np.vstack(dx_labels_list)
    c_preds   = np.vstack(concept_probs_list)
    c_labels  = np.vstack(concept_labels_list)

    dx_f1      = f1_score(dx_labels, (dx_preds > 0.5).astype(int), average="macro", zero_division=0)
    concept_f1 = f1_score(c_labels,  (c_preds  > 0.5).astype(int), average="macro", zero_division=0)

    return {
        "loss"        : total_loss / n,
        "dx_f1"       : float(dx_f1),
        "concept_f1"  : float(concept_f1),
        "loss_dx"     : loss_acc["dx"]      / n,
        "loss_align"  : loss_acc["align"]   / n,
        "loss_concept": loss_acc["concept"] / n,
    }


# ============================================================================
# PHASE 3
# ============================================================================

def evaluate_phase3(
    model,
    dataloader,
    criterion: MultiObjectiveLoss,
    device: torch.device,
    concept_embeddings: torch.Tensor,
    use_rag: bool = True,
) -> dict:
    """
    Evaluate Phase 3 (ShifaMindPhase3RAG).

    concept_embeddings: [num_concepts, 768] detached tensor (frozen from Phase 2).
    """
    model.eval()
    dx_probs_list, dx_labels_list = [], []
    val_losses: list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            dx_labels      = batch["labels"].to(device, non_blocking=True)
            concept_labels = batch["concept_labels"].to(device, non_blocking=True)
            texts          = batch["text"]

            outputs = model(
                input_ids, attention_mask, concept_embeddings,
                input_texts=texts, use_rag=use_rag,
            )
            loss, _ = criterion(outputs, dx_labels, concept_labels)
            val_losses.append(loss.item())

            dx_probs_list.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            dx_labels_list.append(dx_labels.cpu().numpy())

    dx_preds  = np.vstack(dx_probs_list)
    dx_labels = np.vstack(dx_labels_list)

    dx_f1       = f1_score(dx_labels, (dx_preds > 0.5).astype(int), average="macro", zero_division=0)
    dx_micro    = f1_score(dx_labels, (dx_preds > 0.5).astype(int), average="micro", zero_division=0)
    precision   = precision_score(dx_labels, (dx_preds > 0.5).astype(int), average="macro", zero_division=0)
    recall      = recall_score(dx_labels,    (dx_preds > 0.5).astype(int), average="macro", zero_division=0)

    return {
        "loss"     : float(np.mean(val_losses)),
        "macro_f1" : float(dx_f1),
        "micro_f1" : float(dx_micro),
        "precision": float(precision),
        "recall"   : float(recall),
    }
