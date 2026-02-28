"""
graph/build.py

Builds the UMLS knowledge graph used in Phase 2.

Pipeline:
  1. load_umls_cui_mappings  — parse MRCONSO.RRF → concept/ICD-10 → CUI dict
  2. load_umls_relationships — parse MRREL.RRF   → (CUI, rel, CUI) triples
  3. build_umls_graph        — construct NetworkX DiGraph with weighted edges
  4. nx_to_pyg               — compute BioClinicalBERT node features,
                               convert to torch_geometric.data.Data
  5. build_and_save_graph    — orchestrate steps 1-4 and persist to disk

All output files are written to paths defined in config.
"""
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from tqdm.auto import tqdm

import config
from utils.logging_utils import get_logger

log = get_logger()


# ============================================================================
# STEP 1 — UMLS CUI mappings
# ============================================================================

def load_umls_cui_mappings(
    umls_path: Path,
    concepts: List[str],
    icd_codes: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse MRCONSO.RRF and build two dicts:
        concept_to_cui : e.g. {"fever": "C0015967"}
        icd_to_cui     : e.g. {"I50": "C0018802"}

    Only English entries are considered.
    ICD-10 dots are stripped (I10.0 → I10).
    """
    mrconso = umls_path / "MRCONSO.RRF"
    if not mrconso.exists():
        log.error(f"MRCONSO.RRF not found at {mrconso}")
        return {}, {}

    concept_set = set(c.lower() for c in concepts)
    icd_set     = set(icd_codes)

    concept_to_cui: Dict[str, str] = {}
    icd_to_cui:     Dict[str, str] = {}

    t0 = time.time()
    n  = 0

    with open(mrconso, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue

            cui      = parts[0]
            lang     = parts[1]
            source   = parts[11]   # SAB
            term     = parts[14].lower().strip()

            if lang != "ENG":
                continue

            # Concept keywords
            if term in concept_set and term not in concept_to_cui:
                concept_to_cui[term] = cui

            # ICD-10 codes
            if source == "ICD10CM":
                code = parts[13].replace(".", "")
                if code in icd_set and code not in icd_to_cui:
                    icd_to_cui[code] = cui

            n += 1
            if n % 500_000 == 0:
                log.info(f"  MRCONSO: processed {n:,} entries …")

    elapsed = time.time() - t0
    log.info(
        f"MRCONSO done in {elapsed:.0f}s — "
        f"concepts {len(concept_to_cui)}/{len(concepts)}, "
        f"ICD codes {len(icd_to_cui)}/{len(icd_codes)}"
    )
    return concept_to_cui, icd_to_cui


# ============================================================================
# STEP 2 — UMLS relationships
# ============================================================================

def load_umls_relationships(
    umls_path: Path,
    valid_cuis: set,
) -> List[Tuple[str, str, str]]:
    """
    Parse MRREL.RRF and return triples (CUI1, rel, CUI2) where both CUIs
    are in *valid_cuis* and the relationship type is clinically meaningful.
    """
    mrrel = umls_path / "MRREL.RRF"
    if not mrrel.exists():
        log.error(f"MRREL.RRF not found at {mrrel}")
        return []

    # Relationship types that carry clinical signal
    keep_rels = {"CHD", "PAR", "RB", "RN", "SY", "isa"}
    triples: List[Tuple[str, str, str]] = []
    n = 0

    with open(mrrel, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue

            cui1, rel, cui2 = parts[0], parts[3], parts[4]
            if cui1 in valid_cuis and cui2 in valid_cuis and rel in keep_rels:
                triples.append((cui1, rel, cui2))

            n += 1
            if n % 1_000_000 == 0:
                log.info(f"  MRREL: processed {n:,} entries …")

    log.info(f"MRREL done — {len(triples):,} relevant triples found")
    return triples


# ============================================================================
# STEP 3 — NetworkX graph
# ============================================================================

def build_umls_graph(
    concepts: List[str],
    icd_codes: List[str],
    concept_to_cui: Dict[str, str],
    icd_to_cui: Dict[str, str],
    relationships: List[Tuple[str, str, str]],
) -> nx.DiGraph:
    """
    Build a directed, weighted NetworkX graph.

    Nodes:
        concept   nodes — one per clinical keyword
        diagnosis nodes — one per Top-50 ICD-10 code

    Edges:
        UMLS relationship edges (weighted by type)
        ICD chapter similarity edges (weight = 0.3, bidirectional)
    """
    G = nx.DiGraph()

    for c in concepts:
        G.add_node(c, node_type="concept", cui=concept_to_cui.get(c))
    for code in icd_codes:
        G.add_node(code, node_type="diagnosis", cui=icd_to_cui.get(code))

    # CUI → list of nodes (a CUI can map to both a concept and a code)
    cui_to_nodes: Dict[str, List[str]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        if data.get("cui"):
            cui_to_nodes[data["cui"]].append(node)

    # Edge weights by relationship type
    rel_weight = {
        "CHD": 1.0, "PAR": 1.0, "isa": 1.0,  # hierarchical — strongest
        "RB":  0.8, "RN":  0.8,                # broader / narrower semantic
        "SY":  0.5,                             # synonym
    }

    umls_edges = 0
    for cui1, rel, cui2 in relationships:
        for n1 in cui_to_nodes.get(cui1, []):
            for n2 in cui_to_nodes.get(cui2, []):
                if n1 != n2 and not G.has_edge(n1, n2):
                    G.add_edge(n1, n2, edge_type=rel, weight=rel_weight.get(rel, 0.5))
                    umls_edges += 1

    # Chapter-similarity edges between ICD codes that share a letter prefix
    chapter_groups: Dict[str, List[str]] = defaultdict(list)
    for code in icd_codes:
        chapter_groups[code[0]].append(code)

    chapter_edges = 0
    for codes in chapter_groups.values():
        for i, c1 in enumerate(codes):
            for c2 in codes[i + 1:]:
                if not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, edge_type="same_chapter", weight=0.3)
                    G.add_edge(c2, c1, edge_type="same_chapter", weight=0.3)
                    chapter_edges += 2

    log.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"({umls_edges} UMLS + {chapter_edges} chapter-sim)"
    )
    return G


# ============================================================================
# STEP 4 — Convert to PyTorch Geometric Data
# ============================================================================

def nx_to_pyg(G: nx.DiGraph, tokenizer, bert_model, device: torch.device):
    """
    Compute BioClinicalBERT [CLS] embeddings for every node and return a
    torch_geometric.data.Data object.

    Node feature matrix  x : [N, 768]
    edge_index            : [2, E]
    edge_attr             : [E, 1]  (edge weight)

    Custom attributes stored on the Data object:
        node_type_mask   : [N]  (0 = diagnosis, 1 = concept)
        node_to_idx      : dict
        idx_to_node      : dict
    """
    from torch_geometric.data import Data

    all_nodes   = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    # --- Node features via BioClinicalBERT -----------------------------------
    log.info(f"Computing BioClinicalBERT embeddings for {len(all_nodes)} nodes …")
    bert_model.eval()
    node_features = {}

    for node in tqdm(all_nodes, desc="Node embeddings"):
        if G.nodes[node]["node_type"] == "concept":
            text = node
        else:
            text = f"ICD-10 diagnosis code {node}"

        enc = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = bert_model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
        node_features[node] = emb

    x = torch.stack([node_features[n] for n in all_nodes])  # [N, 768]

    # --- Edge index & attributes ---------------------------------------------
    edge_index_list, edge_attr_list = [], []
    for u, v, data in G.edges(data=True):
        edge_index_list.append([node_to_idx[u], node_to_idx[v]])
        edge_attr_list.append(data.get("weight", 1.0))

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float).unsqueeze(-1)

    node_type_mask = torch.tensor(
        [0 if G.nodes[n]["node_type"] == "diagnosis" else 1 for n in all_nodes],
        dtype=torch.long,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_type_mask = node_type_mask
    data.node_to_idx    = node_to_idx
    data.idx_to_node    = {i: n for n, i in node_to_idx.items()}

    log.info(
        f"PyG Data: {data.x.shape[0]} nodes × {data.x.shape[1]}-dim, "
        f"{data.edge_index.shape[1]} edges"
    )
    return data


# ============================================================================
# STEP 5 — Orchestrator
# ============================================================================

def build_and_save_graph(
    tokenizer,
    bert_model,
    device: torch.device,
    concepts: Optional[List[str]] = None,
    icd_codes: Optional[List[str]] = None,
) -> None:
    """
    Full pipeline: UMLS → NetworkX → PyG → disk.

    Skips graph building if graph_data.pt already exists.
    Call this once from scripts/phase2_train.py.
    """
    if config.GRAPH_DATA_PT.exists():
        log.info(f"Graph already exists at {config.GRAPH_DATA_PT} — skipping build.")
        return

    if concepts is None:
        concepts = config.GLOBAL_CONCEPTS
    if icd_codes is None:
        raise ValueError("icd_codes must be provided (Top-50 ICD-10 codes).")

    log.info("=" * 60)
    log.info("Building UMLS knowledge graph …")
    log.info("=" * 60)

    concept_to_cui, icd_to_cui = load_umls_cui_mappings(
        config.UMLS_DIR, concepts, icd_codes
    )
    all_cuis    = set(concept_to_cui.values()) | set(icd_to_cui.values())
    triples     = load_umls_relationships(config.UMLS_DIR, all_cuis)
    nx_graph    = build_umls_graph(concepts, icd_codes, concept_to_cui, icd_to_cui, triples)

    # Persist NetworkX graph (for inspection / debugging)
    with open(config.UMLS_GRAPH_GPICKLE, "wb") as f:
        pickle.dump(nx_graph, f)
    log.info(f"NetworkX graph saved → {config.UMLS_GRAPH_GPICKLE.name}")

    # Convert to PyG and save
    graph_data = nx_to_pyg(nx_graph, tokenizer, bert_model, device)
    torch.save(graph_data, config.GRAPH_DATA_PT)
    log.info(f"PyG graph saved      → {config.GRAPH_DATA_PT.name}")
