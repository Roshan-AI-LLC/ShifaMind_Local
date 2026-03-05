"""
graph/build.py  —  Multi-Source Clinical Knowledge Graph

Replaces the old UMLS-only graph with a four-source KG:

  1. UMLS ontological edges     (structural medical prior)
  2. Co-coding edges            (concept→code, data-driven NPMI from training labels)
  3. Competitive coding edges   (code↔code, mutual exclusivity from training labels)
  4. PubMed co-mention edges    (concept↔concept, literature signal)  [optional]

Edge weights use non-overlapping scalar ranges so the GAT attention
mechanism can distinguish edge types through learned weighting:

  UMLS hierarchical  : 1.00  (CHD, PAR, isa)
  UMLS semantic      : 0.80  (RB, RN)
  UMLS synonym       : 0.60  (SY)
  Co-coding          : 0.65–0.75  (NPMI-scaled, concept→code)
  PubMed co-mention  : 0.45–0.55  (NPMI-scaled, concept↔concept)
  Competitive        : 0.25–0.35  (exclusivity-scaled, code↔code)

Nodes are ordered: concept nodes [0 .. N_concepts-1]
                   diagnosis nodes [N_concepts .. N_concepts+N_codes-1]
This ordering is relied upon by ShifaMindPhase2GAT — do not change it.

Public API (unchanged from v1):
    build_and_save_graph(tokenizer, bert_model, device, concepts, icd_codes,
                         train_concept_labels, train_dx_labels)
"""
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

import config
from utils.logging_utils import get_logger

log = get_logger()

# Graph format version — used to detect and auto-rebuild stale v1 graphs
_GRAPH_VERSION = "v2_multi_source"


# ============================================================================
# HELPERS
# ============================================================================

def _read_env_key(key: str) -> Optional[str]:
    """Read *key* from the environment or from the project .env file."""
    value = os.environ.get(key)
    if value:
        return value
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ============================================================================
# MAIN BUILDER CLASS
# ============================================================================

class MultiSourceKGBuilder:
    """
    Builds a directed, weighted knowledge graph from four complementary
    sources and exposes a to_pyg() method that converts it to a
    torch_geometric.data.Data object suitable for Phase 2 training.

    Parameters
    ----------
    concepts              : ordered list of clinical concept terms (111)
    icd_codes             : ordered list of Top-50 ICD-10 codes (50)
    train_concept_labels  : [N_train, len(concepts)]  binary presence matrix
    train_dx_labels       : [N_train, len(icd_codes)] binary label matrix
    """

    def __init__(
        self,
        concepts: List[str],
        icd_codes: List[str],
        train_concept_labels: np.ndarray,
        train_dx_labels: np.ndarray,
    ) -> None:
        self.concepts = concepts
        self.icd_codes = icd_codes
        self.concept_presence = train_concept_labels.astype(np.float32)  # [N, 111]
        self.dx_labels        = train_dx_labels.astype(np.float32)       # [N, 50]
        self.N_train          = len(train_concept_labels)

        self.G: nx.DiGraph = nx.DiGraph()

        # Populated during UMLS parsing
        self.concept_to_cui: Dict[str, str] = {}
        self.icd_to_cui:     Dict[str, str] = {}
        self.icd_to_name:    Dict[str, str] = {}  # ICD code → human-readable name

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    def build(self) -> nx.DiGraph:
        log.info("=" * 60)
        log.info("Building Multi-Source Clinical Knowledge Graph")
        log.info("=" * 60)

        self._add_nodes()
        self._load_umls_mappings()
        self._add_umls_edges()
        self._add_co_coding_edges()
        self._add_competitive_edges()

        if config.USE_PUBMED_EDGES:
            try:
                self._add_pubmed_edges()
            except Exception as exc:
                log.warning(f"PubMed edge building failed — skipping. Reason: {exc}")

        self._log_stats()
        return self.G

    # ------------------------------------------------------------------
    # STEP 1 — Nodes
    # ------------------------------------------------------------------

    def _add_nodes(self) -> None:
        # Concepts MUST come first: indices [0 .. N_concepts-1]
        for c in self.concepts:
            self.G.add_node(c, node_type="concept")
        # Diagnoses follow: indices [N_concepts .. N_concepts+N_codes-1]
        for code in self.icd_codes:
            self.G.add_node(code, node_type="diagnosis")

        log.info(
            f"Nodes added: {len(self.concepts)} concepts + "
            f"{len(self.icd_codes)} diagnoses = {self.G.number_of_nodes()} total"
        )

    # ------------------------------------------------------------------
    # STEP 2 — UMLS CUI mappings
    # ------------------------------------------------------------------

    def _load_umls_mappings(self) -> None:
        """
        Parse MRCONSO.RRF once and build:
            concept_to_cui : {"fever": "C0015967", ...}
            icd_to_cui     : {"I50": "C0018802", ...}
            icd_to_name    : {"I50": "Heart failure, unspecified", ...}
        """
        mrconso = config.UMLS_MRCONSO
        if not mrconso.exists():
            log.error(
                f"MRCONSO.RRF not found at:\n  {mrconso}\n"
                "Google Drive may not be mounted or the file is cloud-only.\n"
                "Make the file available offline, then delete graph_data.pt to rebuild."
            )
            return

        concept_set = {c.lower() for c in self.concepts}
        icd_set     = set(self.icd_codes)

        t0 = time.time()
        n  = 0

        with open(mrconso, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue

                cui    = parts[0]
                lang   = parts[1]
                source = parts[11]
                term   = parts[14].lower().strip()

                if lang != "ENG":
                    continue

                if term in concept_set and term not in self.concept_to_cui:
                    self.concept_to_cui[term] = cui

                if source == "ICD10CM":
                    code = parts[13].replace(".", "")
                    if code in icd_set and code not in self.icd_to_cui:
                        self.icd_to_cui[code]  = cui
                        self.icd_to_name[code] = parts[14]  # keep original case

                n += 1
                if n % 500_000 == 0:
                    log.info(f"  MRCONSO: {n:,} entries scanned …")

        elapsed = time.time() - t0
        log.info(
            f"MRCONSO done in {elapsed:.0f}s — "
            f"concepts {len(self.concept_to_cui)}/{len(self.concepts)}, "
            f"ICD codes {len(self.icd_to_cui)}/{len(self.icd_codes)}"
        )

        # Attach CUI and name to graph nodes
        for node, data in self.G.nodes(data=True):
            if data["node_type"] == "concept":
                data["cui"] = self.concept_to_cui.get(node.lower())
            else:
                data["cui"]  = self.icd_to_cui.get(node)
                data["name"] = self.icd_to_name.get(node, node)

    # ------------------------------------------------------------------
    # STEP 3 — UMLS relationship edges
    # ------------------------------------------------------------------

    def _add_umls_edges(self) -> None:
        """Parse MRREL.RRF and add weighted ontological edges."""
        # Reset counter — used later by to_pyg() and cache-invalidation logic
        self.umls_edge_count: int = 0

        mrrel = config.UMLS_MRREL
        if not mrrel.exists():
            log.error(
                f"MRREL.RRF not found at:\n  {mrrel}\n"
                "Google Drive may not be mounted or the file is cloud-only.\n"
                "Make the file available offline, then delete graph_data.pt to rebuild."
            )
            return

        keep_rels = {"CHD", "PAR", "isa", "RB", "RN", "SY"}
        rel_weight = {
            "CHD": 1.0, "PAR": 1.0, "isa": 1.0,
            "RB":  0.8, "RN":  0.8,
            "SY":  0.6,
        }

        # CUI → graph nodes
        cui_to_nodes: Dict[str, List[str]] = defaultdict(list)
        for node, data in self.G.nodes(data=True):
            if data.get("cui"):
                cui_to_nodes[data["cui"]].append(node)
        valid_cuis = set(cui_to_nodes.keys())

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
                    log.info(f"  MRREL: {n:,} entries scanned …")

        log.info(f"MRREL done — {len(triples):,} relevant triples")

        umls_edges = 0
        for cui1, rel, cui2 in triples:
            w = rel_weight.get(rel, 0.6)
            for n1 in cui_to_nodes.get(cui1, []):
                for n2 in cui_to_nodes.get(cui2, []):
                    if n1 != n2 and not self.G.has_edge(n1, n2):
                        self.G.add_edge(n1, n2, edge_type=f"umls_{rel}", weight=w)
                        umls_edges += 1

        self.umls_edge_count = umls_edges
        log.info(f"UMLS edges added: {umls_edges:,}")

    # ------------------------------------------------------------------
    # STEP 4 — Co-coding edges  (concept → code, data-driven)
    # ------------------------------------------------------------------

    def _add_co_coding_edges(self) -> None:
        """
        For each (concept_i, code_j) pair compute NPMI over training data:

            NPMI(concept, code) = PMI(concept, code) / -log P(concept ∩ code)

        Edge added when NPMI > GRAPH_COCODING_NPMI_MIN.
        Weight scaled to [0.65, 0.75].

        Medical insight: captures which symptom/finding patterns are
        SPECIFICALLY associated with a code in actual coding practice,
        not just general medical co-occurrence.
        """
        N   = self.N_train
        eps = 1e-10

        # Marginal counts
        concept_counts = self.concept_presence.sum(axis=0)  # [111]
        code_counts    = self.dx_labels.sum(axis=0)          # [50]

        # Joint counts [111, 50] via matrix multiply
        joint = self.concept_presence.T @ self.dx_labels     # [111, 50]

        edges_added = 0
        for ci, concept in enumerate(self.concepts):
            p_c = float(concept_counts[ci]) / N
            if p_c < eps:
                continue
            for di, code in enumerate(self.icd_codes):
                p_d  = float(code_counts[di]) / N
                p_cd = float(joint[ci, di]) / N
                if p_d < eps or p_cd < eps:
                    continue

                pmi  = np.log(p_cd / (p_c * p_d + eps))
                npmi = pmi / (-np.log(p_cd + eps))

                if npmi > config.GRAPH_COCODING_NPMI_MIN:
                    w = 0.65 + 0.10 * min(float(npmi), 1.0)
                    if not self.G.has_edge(concept, code):
                        self.G.add_edge(
                            concept, code,
                            edge_type="co_coding",
                            weight=round(w, 4),
                        )
                        edges_added += 1

        log.info(f"Co-coding edges added: {edges_added:,}")

    # ------------------------------------------------------------------
    # STEP 5 — Competitive coding edges  (code ↔ code, data-driven)
    # ------------------------------------------------------------------

    def _add_competitive_edges(self) -> None:
        """
        For each pair (code_i, code_j) compute an exclusivity score:

            n_either   = notes where at least one is coded
            n_only_one = n_either - 2 × n_both
            exclusivity = n_only_one / n_either

        High exclusivity → codes are alternatives (coding one suppresses
        the other) — this is the zero-sum structure of coding specificity.

        Edge added when exclusivity > GRAPH_COMPETITIVE_MIN and both
        codes appear at least GRAPH_COMPETITIVE_MIN_FREQ times.
        Weight scaled to [0.25, 0.35].

        Medical insight: distinguishes "Type 1 diabetes" from
        "Type 2 diabetes", "Sepsis" from "Bacteremia", etc. — pairs
        the model needs to discriminate, not merge.
        """
        min_freq = config.GRAPH_COMPETITIVE_MIN_FREQ
        code_counts = self.dx_labels.sum(axis=0)        # [50]
        joint       = self.dx_labels.T @ self.dx_labels  # [50, 50]  n_both for each pair

        edges_added = 0
        threshold   = config.GRAPH_COMPETITIVE_MIN

        for i, code_i in enumerate(self.icd_codes):
            if code_counts[i] < min_freq:
                continue
            for j, code_j in enumerate(self.icd_codes):
                if j <= i:
                    continue
                if code_counts[j] < min_freq:
                    continue

                n_both   = float(joint[i, j])
                n_either = float(code_counts[i] + code_counts[j] - n_both)
                if n_either < 1:
                    continue

                n_only_one  = float(code_counts[i] + code_counts[j]) - 2.0 * n_both
                exclusivity = n_only_one / n_either

                if exclusivity > threshold:
                    # Scale into [0.25, 0.35]
                    span = 1.0 - threshold
                    w    = 0.25 + 0.10 * min((exclusivity - threshold) / span, 1.0)
                    w    = round(w, 4)
                    self.G.add_edge(code_i, code_j, edge_type="competitive", weight=w)
                    self.G.add_edge(code_j, code_i, edge_type="competitive", weight=w)
                    edges_added += 2

        log.info(f"Competitive edges added: {edges_added:,}")

    # ------------------------------------------------------------------
    # STEP 6 — PubMed co-mention edges  (concept ↔ concept, literature)
    # ------------------------------------------------------------------

    def _add_pubmed_edges(self) -> None:
        """
        Fetch PubMed abstracts for each ICD-10 condition name, detect
        concept term co-mentions, compute NPMI, and add edges between
        concept pairs with NPMI > GRAPH_PUBMED_NPMI_MIN.

        Abstracts are cached to PUBMED_CACHE_JSON — the potentially slow
        network fetch only runs once.  Weight scaled to [0.45, 0.55].

        Medical insight: adds literature evidence about which clinical
        concepts are studied together — enriches the graph with
        knowledge that doesn't appear in UMLS or training data.
        """
        abstracts = self._fetch_and_cache_abstracts()
        if not abstracts:
            log.warning("No PubMed abstracts retrieved — skipping PubMed edges.")
            return

        log.info(f"Computing co-mention NPMI over {len(abstracts):,} abstracts …")
        concepts_lower = [c.lower() for c in self.concepts]
        n_docs = len(abstracts)

        # Binary presence matrix [n_docs, n_concepts]
        presence = np.zeros((n_docs, len(self.concepts)), dtype=np.int8)
        for idx, text in enumerate(tqdm(abstracts, desc="Concept detection", leave=False)):
            text_lower = text.lower()
            for ci, concept in enumerate(concepts_lower):
                if concept in text_lower:
                    presence[idx, ci] = 1

        concept_counts = presence.sum(axis=0)               # [111]
        joint          = presence.T @ presence               # [111, 111]
        eps            = 1e-10

        edges_added = 0
        for i in range(len(self.concepts)):
            if concept_counts[i] < 5:
                continue
            for j in range(i + 1, len(self.concepts)):
                if concept_counts[j] < 5:
                    continue

                p_i  = float(concept_counts[i]) / n_docs
                p_j  = float(concept_counts[j]) / n_docs
                p_ij = float(joint[i, j]) / n_docs
                if p_ij < eps:
                    continue

                pmi  = np.log(p_ij / (p_i * p_j + eps))
                npmi = pmi / (-np.log(p_ij + eps))

                if npmi > config.GRAPH_PUBMED_NPMI_MIN:
                    w = round(0.45 + 0.10 * min(float(npmi), 1.0), 4)
                    ci_name = self.concepts[i]
                    cj_name = self.concepts[j]
                    if not self.G.has_edge(ci_name, cj_name):
                        self.G.add_edge(ci_name, cj_name, edge_type="pubmed", weight=w)
                        edges_added += 1
                    if not self.G.has_edge(cj_name, ci_name):
                        self.G.add_edge(cj_name, ci_name, edge_type="pubmed", weight=w)
                        edges_added += 1

        log.info(f"PubMed co-mention edges added: {edges_added:,}")

    def _fetch_and_cache_abstracts(self) -> List[str]:
        """
        Return a flat list of abstract texts.  On first call the texts are
        fetched from NCBI and cached to PUBMED_CACHE_JSON; subsequent calls
        load from the cache file immediately.
        """
        cache_path = config.PUBMED_CACHE_JSON
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            log.info(
                f"Loaded {len(cached['abstracts']):,} PubMed abstracts from cache "
                f"({cache_path.name})"
            )
            return cached["abstracts"]

        try:
            from Bio import Entrez
        except ImportError:
            raise RuntimeError(
                "biopython is required for PubMed edges. "
                "Install with: pip install biopython"
            )

        api_key = _read_env_key("NCBI_API_KEY")
        Entrez.email   = "shifamind@research.ai"
        if api_key:
            Entrez.api_key = api_key
            rate_delay     = 0.11   # ~9 req/sec, safely under 10
        else:
            rate_delay = 0.35       # ~3 req/sec, safely under 3
            log.warning(
                "No NCBI_API_KEY found — using 3 req/sec. "
                "Set NCBI_API_KEY in .env for faster fetching."
            )

        all_abstracts: List[str] = []
        per_code = config.PUBMED_ABSTRACTS_PER_CODE

        for code in tqdm(self.icd_codes, desc="Fetching PubMed"):
            name = self.icd_to_name.get(code, code)
            # Build a query targeting clinical literature about this condition
            query = f'"{name}"[Title/Abstract] AND "clinical"[Title/Abstract]'

            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=per_code)
                record = Entrez.read(handle)
                handle.close()
                pmids  = record.get("IdList", [])

                if pmids:
                    time.sleep(rate_delay)
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=",".join(pmids),
                        rettype="abstract",
                        retmode="text",
                    )
                    text = handle.read()
                    handle.close()
                    all_abstracts.extend(text.split("\n\n"))  # split into paragraphs

                time.sleep(rate_delay)

            except Exception as exc:
                log.warning(f"  PubMed fetch failed for {code} ({name}): {exc}")
                continue

        # Persist cache
        with open(cache_path, "w") as f:
            json.dump(
                {"abstracts": all_abstracts, "n_codes": len(self.icd_codes)},
                f,
            )
        log.info(f"Fetched {len(all_abstracts):,} PubMed abstract paragraphs → cached")
        return all_abstracts

    # ------------------------------------------------------------------
    # STATS LOGGING
    # ------------------------------------------------------------------

    def _log_stats(self) -> None:
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for _, _, data in self.G.edges(data=True):
            edge_type_counts[data.get("edge_type", "unknown")] += 1

        stats = {
            "graph_version"  : _GRAPH_VERSION,
            "n_nodes"        : self.G.number_of_nodes(),
            "n_concepts"     : len(self.concepts),
            "n_diagnoses"    : len(self.icd_codes),
            "n_edges_total"  : self.G.number_of_edges(),
            "edge_type_counts": dict(edge_type_counts),
        }

        log.info("=" * 60)
        log.info("Multi-Source KG Summary")
        log.info("=" * 60)
        log.info(f"  Nodes        : {stats['n_nodes']:>6}  ({stats['n_concepts']} concepts + {stats['n_diagnoses']} diagnoses)")
        for etype, cnt in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {etype:<22}: {cnt:>6,}")
        log.info(f"  {'Total edges':<22}: {stats['n_edges_total']:>6,}")
        log.info("=" * 60)

        stats_path = config.GRAPH_P2 / "graph_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # ------------------------------------------------------------------
    # CONVERT TO PYTORCH GEOMETRIC
    # ------------------------------------------------------------------

    def to_pyg(
        self,
        tokenizer,
        bert_model,
        device: torch.device,
    ):
        """
        Compute BioClinicalBERT [CLS] embeddings for every node and return a
        torch_geometric.data.Data object.

        Diagnosis nodes use the UMLS condition name for a richer embedding:
            "Heart failure, unspecified — ICD-10 I50"
        instead of the old generic "ICD-10 diagnosis code I50".

        Returns
        -------
        data : torch_geometric.data.Data
            x               [N, 768]   node features
            edge_index      [2, E]     directed edges
            edge_attr       [E, 1]     scalar edge weights
            node_type_mask  [N]        0=diagnosis, 1=concept
            node_to_idx     dict
            idx_to_node     dict
            graph_version   str        version tag for cache invalidation
        """
        from torch_geometric.data import Data

        all_nodes   = list(self.G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        # --- Node features -----------------------------------------------
        log.info(f"Computing BioClinicalBERT embeddings for {len(all_nodes)} nodes …")
        bert_model.eval()
        node_features: Dict[str, torch.Tensor] = {}

        for node in tqdm(all_nodes, desc="Node embeddings"):
            ndata = self.G.nodes[node]
            if ndata["node_type"] == "concept":
                text = node
            else:
                name = ndata.get("name", node)
                text = f"{name} — ICD-10 {node}"

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

        # --- Edge index & attributes -------------------------------------
        src_list, dst_list, weight_list = [], [], []
        for u, v, edata in self.G.edges(data=True):
            src_list.append(node_to_idx[u])
            dst_list.append(node_to_idx[v])
            weight_list.append(edata.get("weight", 1.0))

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        ).contiguous()
        edge_attr  = torch.tensor(weight_list, dtype=torch.float).unsqueeze(-1)

        node_type_mask = torch.tensor(
            [0 if self.G.nodes[n]["node_type"] == "diagnosis" else 1
             for n in all_nodes],
            dtype=torch.long,
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.node_type_mask  = node_type_mask
        data.node_to_idx     = node_to_idx
        data.idx_to_node     = {i: n for n, i in node_to_idx.items()}
        data.graph_version   = _GRAPH_VERSION
        data.umls_edge_count = getattr(self, "umls_edge_count", 0)

        log.info(
            f"PyG Data: {data.x.shape[0]} nodes × {data.x.shape[1]}-dim, "
            f"{data.edge_index.shape[1]} edges"
        )
        return data


# ============================================================================
# PUBLIC API  (called from phase2_train.py — signature backward-compatible)
# ============================================================================

def build_and_save_graph(
    tokenizer,
    bert_model,
    device: torch.device,
    concepts: Optional[List[str]]   = None,
    icd_codes: Optional[List[str]]  = None,
    train_concept_labels: Optional[np.ndarray] = None,
    train_dx_labels:      Optional[np.ndarray] = None,
) -> None:
    """
    Full pipeline: multi-source KG → NetworkX → PyG → disk.

    Skips if graph_data.pt already exists AND was built with the current
    multi-source pipeline (detected via graph_version attribute).
    Old v1 graphs are automatically rebuilt.

    Parameters
    ----------
    tokenizer, bert_model, device
        Used only for computing BioClinicalBERT node embeddings.
    concepts
        Clinical concept terms.  Defaults to config.GLOBAL_CONCEPTS.
    icd_codes
        Top-50 ICD-10 codes.  Required.
    train_concept_labels
        Binary concept presence matrix [N_train, n_concepts].  Required
        for data-driven edges (co-coding, competitive).
    train_dx_labels
        Binary diagnosis label matrix [N_train, n_codes].  Required.
    """
    if concepts is None:
        concepts = config.GLOBAL_CONCEPTS
    if icd_codes is None:
        raise ValueError("icd_codes must be provided.")
    if train_concept_labels is None or train_dx_labels is None:
        raise ValueError(
            "train_concept_labels and train_dx_labels are required "
            "for the multi-source KG.  Pass them from phase2_train.py."
        )

    # Cache-hit check: skip only if saved graph is current AND complete
    if config.GRAPH_DATA_PT.exists():
        try:
            existing = torch.load(
                config.GRAPH_DATA_PT, map_location="cpu", weights_only=False
            )
            if getattr(existing, "graph_version", "v1") == _GRAPH_VERSION:
                umls_was_missing   = getattr(existing, "umls_edge_count", 0) == 0
                umls_now_available = config.UMLS_MRCONSO.exists()
                if umls_was_missing and umls_now_available:
                    log.info(
                        "Cached graph was built WITHOUT UMLS (Google Drive was offline). "
                        "UMLS is now available — rebuilding to include ontological edges."
                    )
                    # fall through to rebuild
                else:
                    if umls_was_missing:
                        log.warning(
                            "Using cached graph built WITHOUT UMLS edges. "
                            "Make MRCONSO.RRF available offline and delete "
                            f"{config.GRAPH_DATA_PT.name} to trigger a rebuild."
                        )
                    log.info(
                        f"Graph cached ({config.GRAPH_DATA_PT.name}) — skipping build."
                    )
                    return
            else:
                log.info("Stale v1 graph detected — rebuilding with multi-source pipeline.")
        except Exception:
            log.info("Could not read existing graph — rebuilding.")

    log.info("=" * 60)
    log.info("Multi-Source KG Build")
    log.info("=" * 60)

    builder   = MultiSourceKGBuilder(
        concepts=concepts,
        icd_codes=icd_codes,
        train_concept_labels=train_concept_labels,
        train_dx_labels=train_dx_labels,
    )
    nx_graph  = builder.build()

    # Persist NetworkX graph (for inspection / debugging)
    with open(config.UMLS_GRAPH_GPICKLE, "wb") as f:
        pickle.dump(nx_graph, f)
    log.info(f"NetworkX graph saved → {config.UMLS_GRAPH_GPICKLE.name}")

    # Convert to PyG and save
    graph_data = builder.to_pyg(tokenizer, bert_model, device)
    torch.save(graph_data, config.GRAPH_DATA_PT)
    log.info(f"PyG graph saved      → {config.GRAPH_DATA_PT.name}")
