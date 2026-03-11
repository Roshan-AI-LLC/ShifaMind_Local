"""
benchmark/train_all.py
──────────────────────────────────────────────────────────────────────────────
Train all Group A baselines sequentially on the SAME data splits as ShifaMind.
Each model is saved to:
    benchmark/checkpoints/{model_name}_best.pt

Usage:
    cd ShifaMind_Local
    python benchmark/train_all.py                    # train all models
    python benchmark/train_all.py --models caml laat # train specific models
    python benchmark/train_all.py --skip-trained     # skip already-trained

Evaluation (threshold tuning + test metrics) is handled by evaluate_all.py.
This script only trains and saves best-val-loss checkpoints.
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models.caml        import CAML
from benchmark.models.laat        import LAAT
from benchmark.models.plm_icd     import PLMICD
from benchmark.models.msmn        import MSMN, build_synonym_tensors
from benchmark.models.vanilla_cbm import VanillaCBM


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str = "benchmark/config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> torch.device:
    setting = cfg.get("device", "auto")
    if setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(setting)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(cfg: dict):
    d = cfg["data"]

    def _pkl(key):
        with open(ROOT / d[key], "rb") as f:
            return pickle.load(f)

    train = _pkl("train_split")
    val   = _pkl("val_split")
    test  = _pkl("test_split")

    train_con = np.load(ROOT / d["train_concept_labels"])
    val_con   = np.load(ROOT / d["val_concept_labels"])
    test_con  = np.load(ROOT / d["test_concept_labels"])

    return train, val, test, train_con, val_con, test_con


def _extract(split):
    """Extract texts and labels from a split (list of dicts or DataFrame)."""
    if hasattr(split, "iterrows"):          # DataFrame
        texts  = split["text"].tolist()
        labels = split[[c for c in split.columns if c != "text"]].values.tolist()
    else:                                   # list of dicts
        texts  = [s["text"] for s in split]
        labels = [s["labels"] for s in split]
    return texts, np.array(labels, dtype=np.float32)


class BaselineDataset(Dataset):
    """
    Tokenises clinical notes for all baseline models.
    Returns: input_ids, attention_mask, labels, concept_labels (floats).
    """
    def __init__(self, texts, labels, concept_labels, tokenizer, max_length: int):
        self.texts          = texts
        self.labels         = labels
        self.concept_labels = concept_labels
        self.tokenizer      = tokenizer
        self.max_length     = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding         = "max_length",
            truncation      = True,
            max_length      = self.max_length,
            return_tensors  = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.FloatTensor(self.labels[idx]),
            "concept_labels": torch.FloatTensor(self.concept_labels[idx]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper (val loop)
# ─────────────────────────────────────────────────────────────────────────────

def compute_val_loss(model, loader, criterion, device, model_name: str) -> dict:
    model.eval()
    total_loss = 0.0
    probs_list, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            con   = batch["concept_labels"].to(device)

            if model_name == "vanilla_cbm":
                out   = model(ids, mask)
                loss  = criterion(out["logits"], labs)
                # Stage 1 also adds concept supervision
                if out.get("concept_logits") is not None:
                    loss = loss + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                        out["concept_logits"], con
                    )
            else:
                out  = model(ids, mask)
                loss = criterion(out["logits"], labs)

            total_loss  += loss.item()
            probs_list.append(torch.sigmoid(out["logits"]).cpu().numpy())
            labels_list.append(labs.cpu().numpy())

    from sklearn.metrics import f1_score
    probs  = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds  = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "loss"     : total_loss / len(loader),
        "macro_f1" : macro_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model training routines
# ─────────────────────────────────────────────────────────────────────────────

def _make_loaders(train_ds, val_ds, cfg):
    t = cfg["training"]
    return (
        DataLoader(train_ds, batch_size=t["train_batch_size"],
                   shuffle=True,  num_workers=t["num_workers"]),
        DataLoader(val_ds,   batch_size=t["val_batch_size"],
                   shuffle=False, num_workers=t["num_workers"]),
    )


def _save_checkpoint(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)
    print(f"  Saved → {path.name}  (val macro_f1={meta.get('val_macro_f1', '?'):.4f})")


# ──── CAML / LAAT (non-BERT) ────────────────────────────────────────────────

def train_cnn_model(
    model_name: str,
    model:      nn.Module,
    train_ds,
    val_ds,
    cfg:        dict,
    device:     torch.device,
    ckpt_dir:   Path,
) -> None:
    mcfg    = cfg["group_a"][model_name]
    t       = cfg["training"]
    epochs  = mcfg["epochs"]
    lr      = mcfg["lr"]
    wd      = mcfg.get("weight_decay", 1e-5)

    train_loader, val_loader = _make_loaders(train_ds, val_ds, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1   = -1.0
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}] ep{epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            out  = model(ids, mask,
                         chunk_size=mcfg.get("chunk_size", 512),
                         chunk_overlap=mcfg.get("chunk_overlap", 64))
            loss = criterion(out["logits"], labs)

            if t["grad_accum_steps"] > 1:
                loss = loss / t["grad_accum_steps"]
            loss.backward()

            if step % t["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * t["grad_accum_steps"]
            pbar.set_postfix(loss=f"{train_loss/step:.4f}")

        scheduler.step()
        val_m = compute_val_loss(model, val_loader, criterion, device, model_name)
        print(f"  Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_m['loss']:.4f}  val_macro_f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            _save_checkpoint(model, ckpt_path,
                             {"epoch": epoch, "val_macro_f1": best_f1})


# ──── BERT-based models (PLM-ICD, MSMN) ────────────────────────────────────

def train_bert_model(
    model_name: str,
    model:      nn.Module,
    train_ds,
    val_ds,
    cfg:        dict,
    device:     torch.device,
    ckpt_dir:   Path,
    extra_forward_kwargs: dict = None,
) -> None:
    mcfg    = cfg["group_a"][model_name]
    t       = cfg["training"]
    epochs  = mcfg["epochs"]
    lr_bert = mcfg["lr_bert"]
    lr_head = mcfg["lr_head"]
    freeze_n = mcfg.get("freeze_bert_epochs", 0)

    # Two-group optimizer: BERT backbone at low LR, head at high LR
    bert_params = list(model.bert.parameters())
    bert_ids    = {id(p) for p in bert_params}
    head_params = [p for p in model.parameters() if id(p) not in bert_ids]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr_bert, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 0.0},
    ])
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = _make_loaders(train_ds, val_ds, cfg)
    best_f1   = -1.0
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        # Freeze BERT during warm-up epochs
        requires_grad = (epoch > freeze_n)
        for p in model.bert.parameters():
            p.requires_grad = requires_grad

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}] ep{epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            fwd_kwargs = extra_forward_kwargs or {}
            out  = model(ids, mask, **fwd_kwargs)
            loss = criterion(out["logits"], labs)

            if t["grad_accum_steps"] > 1:
                loss = loss / t["grad_accum_steps"]
            loss.backward()

            if step % t["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * t["grad_accum_steps"]
            pbar.set_postfix(loss=f"{train_loss/step:.4f}")

        scheduler.step()
        val_m = compute_val_loss(model, val_loader, criterion, device, model_name)
        print(f"  Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_m['loss']:.4f}  val_macro_f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            _save_checkpoint(model, ckpt_path,
                             {"epoch": epoch, "val_macro_f1": best_f1})


# ──── Vanilla CBM (two-stage) ────────────────────────────────────────────────

def train_vanilla_cbm(
    model:    VanillaCBM,
    train_ds,
    val_ds,
    cfg:      dict,
    device:   torch.device,
    ckpt_dir: Path,
) -> None:
    mcfg     = cfg["group_a"]["vanilla_cbm"]
    t        = cfg["training"]
    ckpt_path = ckpt_dir / "vanilla_cbm_best.pt"

    concept_criterion = nn.BCEWithLogitsLoss()
    diag_criterion    = nn.BCEWithLogitsLoss()
    train_loader, val_loader = _make_loaders(train_ds, val_ds, cfg)
    best_f1 = -1.0

    # ── Stage 1: BERT + concept_head ──────────────────────────────────────
    print("\n  [vanilla_cbm] Stage 1: training BERT + concept_head …")
    model.unfreeze_all()
    model.diag_head.requires_grad_(False)

    bert_params = list(model.bert.parameters())
    bert_ids    = {id(p) for p in bert_params}
    head_params = [p for p in model.concept_head.parameters()]

    opt_s1 = torch.optim.AdamW([
        {"params": bert_params, "lr": mcfg["lr_stage1"], "weight_decay": 0.01},
        {"params": head_params, "lr": mcfg["lr_stage1"] * 10, "weight_decay": 0.0},
    ])

    for epoch in range(1, mcfg["epochs_stage1"] + 1):
        model.train()
        model.diag_head.requires_grad_(False)
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  [vanilla_cbm] S1 ep{epoch}/{mcfg['epochs_stage1']}")
        for step, batch in enumerate(pbar, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            con  = batch["concept_labels"].to(device)

            out  = model(ids, mask)
            loss = concept_criterion(out["concept_logits"], con)
            (loss / t["grad_accum_steps"]).backward()

            if step % t["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                opt_s1.step()
                opt_s1.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss/step:.4f}")

        print(f"    Stage 1 epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}")

    # ── Stage 2: diagnosis_head only (BERT + concept_head frozen) ─────────
    print("\n  [vanilla_cbm] Stage 2: training diagnosis_head (BERT frozen) …")
    model.freeze_stage1()
    model.diag_head.requires_grad_(True)

    opt_s2 = torch.optim.Adam(model.diag_head.parameters(), lr=mcfg["lr_stage2"])

    for epoch in range(1, mcfg["epochs_stage2"] + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"  [vanilla_cbm] S2 ep{epoch}/{mcfg['epochs_stage2']}")
        for step, batch in enumerate(pbar, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            out  = model(ids, mask)
            loss = diag_criterion(out["logits"], labs)
            (loss / t["grad_accum_steps"]).backward()

            if step % t["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
                opt_s2.step()
                opt_s2.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss/step:.4f}")

        val_m = compute_val_loss(model, val_loader, diag_criterion, device, "vanilla_cbm")
        print(f"    Stage 2 epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_m['loss']:.4f}  val_macro_f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            _save_checkpoint(model, ckpt_path,
                             {"epoch": f"S2-{epoch}", "val_macro_f1": best_f1})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all Group A baselines")
    parser.add_argument("--models",       nargs="+",
                        choices=["caml", "laat", "plm_icd", "msmn", "vanilla_cbm"],
                        default=["caml", "laat", "plm_icd", "msmn", "vanilla_cbm"],
                        help="Which models to train (default: all)")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Skip models that already have a checkpoint")
    parser.add_argument("--config",       default="benchmark/config.yaml")
    args = parser.parse_args()

    cfg      = load_cfg(args.config)
    device   = get_device(cfg)
    seed     = cfg["seed"]
    ckpt_dir = ROOT / cfg["checkpoints"]["baselines_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Device: {device}  |  Seed: {seed}")

    # ── Load data splits ──────────────────────────────────────────────────
    print("Loading data splits …")
    train_split, val_split, test_split, train_con, val_con, test_con = load_splits(cfg)
    train_texts, train_labels = _extract(train_split)
    val_texts,   val_labels   = _extract(val_split)

    num_labels   = train_labels.shape[1]
    num_concepts = train_con.shape[1]
    print(f"Train={len(train_texts)}  Val={len(val_texts)}  "
          f"labels={num_labels}  concepts={num_concepts}")

    # ── BERT tokenizer (shared for all BERT-based models) ─────────────────
    bert_name  = cfg["group_a"]["plm_icd"]["bert_model"]
    print(f"Loading tokenizer: {bert_name} …")
    tokenizer  = AutoTokenizer.from_pretrained(bert_name)
    vocab_size = tokenizer.vocab_size
    max_len    = cfg["training"]["max_length"]

    # ── Build datasets ─────────────────────────────────────────────────────
    train_ds = BaselineDataset(train_texts, train_labels, train_con, tokenizer, max_len)
    val_ds   = BaselineDataset(val_texts,   val_labels,   val_con,   tokenizer, max_len)

    # ─────────────────────────────────────────────────────────────────────
    # Train each model
    # ─────────────────────────────────────────────────────────────────────

    timing = {}

    for model_name in args.models:
        ckpt_path = ckpt_dir / f"{model_name}_best.pt"
        if args.skip_trained and ckpt_path.exists():
            print(f"\n[{model_name}] checkpoint exists — skipping (--skip-trained)")
            continue

        print(f"\n{'='*60}")
        print(f"  Training: {model_name.upper()}")
        print(f"{'='*60}")
        t0 = time.time()

        # ── Build model ─────────────────────────────────────────────────────
        if model_name == "caml":
            mcfg  = cfg["group_a"]["caml"]
            model = CAML(
                vocab_size   = vocab_size,
                num_labels   = num_labels,
                embed_dim    = mcfg["embed_dim"],
                num_filters  = mcfg["num_filters"],
                filter_size  = mcfg["filter_size"],
                dropout      = mcfg["dropout"],
                pad_token_id = tokenizer.pad_token_id or 0,
            ).to(device)
            train_cnn_model("caml", model, train_ds, val_ds, cfg, device, ckpt_dir)

        elif model_name == "laat":
            mcfg  = cfg["group_a"]["laat"]
            model = LAAT(
                vocab_size       = vocab_size,
                num_labels       = num_labels,
                embed_dim        = mcfg["embed_dim"],
                hidden_dim       = mcfg["hidden_dim"],
                label_embed_dim  = mcfg["label_embed_dim"],
                dropout          = mcfg["dropout"],
                pad_token_id     = tokenizer.pad_token_id or 0,
            ).to(device)
            train_cnn_model("laat", model, train_ds, val_ds, cfg, device, ckpt_dir)

        elif model_name == "plm_icd":
            mcfg  = cfg["group_a"]["plm_icd"]
            model = PLMICD(
                bert_model_name = mcfg["bert_model"],
                num_labels      = num_labels,
                hidden_size     = mcfg["hidden_size"],
                dropout         = mcfg["dropout"],
            ).to(device)
            train_bert_model("plm_icd", model, train_ds, val_ds, cfg, device, ckpt_dir)

        elif model_name == "msmn":
            mcfg = cfg["group_a"]["msmn"]

            # Build synonym tensors from top50_icd10_info.json
            with open(ROOT / cfg["data"]["top50_info"]) as f:
                top50_info = json.load(f)
            # top50_codes is the column order of train_labels
            if hasattr(train_split, "columns"):
                top50_codes = [c for c in train_split.columns if c != "text"]
            else:
                top50_codes = list(top50_info.keys())[:num_labels]

            syn_ids, syn_mask = build_synonym_tensors(
                top50_info   = top50_info,
                top50_codes  = top50_codes,
                tokenizer    = tokenizer,
                num_synonyms = mcfg["num_synonyms"],
                max_syn_len  = 32,
                device       = device,
            )

            model = MSMN(
                bert_model_name   = mcfg["bert_model"],
                num_labels        = num_labels,
                num_synonyms      = mcfg["num_synonyms"],
                hidden_size       = mcfg["hidden_size"],
                dropout           = mcfg["dropout"],
                synonym_input_ids = syn_ids,
                synonym_attn_mask = syn_mask,
            ).to(device)

            # Pre-compute synonym CLS embeddings once
            print("  Pre-computing synonym BERT embeddings …")
            model.eval()
            with torch.no_grad():
                synonym_cls = model.encode_synonyms()   # [K, S, H]
            print(f"  Synonym embeddings: {synonym_cls.shape}")

            train_bert_model(
                "msmn", model, train_ds, val_ds, cfg, device, ckpt_dir,
                extra_forward_kwargs={"synonym_cls": synonym_cls},
            )

        elif model_name == "vanilla_cbm":
            mcfg  = cfg["group_a"]["vanilla_cbm"]
            model = VanillaCBM(
                bert_model_name = mcfg["bert_model"],
                num_concepts    = num_concepts,
                num_labels      = num_labels,
                hidden_size     = mcfg["hidden_size"],
                dropout         = mcfg["dropout"],
            ).to(device)
            train_vanilla_cbm(model, train_ds, val_ds, cfg, device, ckpt_dir)

        elapsed = time.time() - t0
        timing[model_name] = elapsed
        print(f"\n  [{model_name}] Training done in {elapsed/3600:.2f}h")

    print("\n" + "="*60)
    print("  All requested models trained.")
    for name, secs in timing.items():
        print(f"    {name:<15} {secs/3600:.2f}h")
    print(f"  Checkpoints → {ckpt_dir}")
    print("  Next: python benchmark/evaluate_all.py")


if __name__ == "__main__":
    main()
