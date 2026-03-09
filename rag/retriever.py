"""
rag/retriever.py

SimpleRAG — FAISS-backed retriever for Phase 3.

Features:
  • Index is serialised to disk (config.FAISS_INDEX) after first build.
  • On subsequent runs the index is loaded from cache — no re-encoding.
  • retrieve() returns a plain string (concatenated relevant passages).
  • retrieve_batch() encodes all queries in ONE call for throughput.
  • Always uses a CPU FAISS index — MPS has no GPU FAISS support.

Evidence corpus sources:
  1. _CLINICAL_KB  — expanded clinical knowledge-base passages covering all
                    ICD-10 chapters present in the Top-50 diagnosis list.
  2. MIMIC prototypes — real clinical note excerpts per diagnosis sampled from
                    the training set (PROTOTYPES_PER_DX per code).
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import config
from utils.logging_utils import get_logger

log = get_logger()


# ============================================================================
# CLINICAL KNOWLEDGE BASE  (expanded — covers all Top-50 ICD-10 chapters)
# Each passage is short, keyword-rich, and uses medical terminology that
# will closely match concept-name RAG queries (e.g. "sepsis hypotension lactate").
# ============================================================================

_CLINICAL_KB = {
    # ── Respiratory ──────────────────────────────────────────────────────────
    "J"   : ("Respiratory: cough dyspnea hypoxia respiratory failure oxygen saturation "
             "pulmonary infiltrate chest imaging bronchospasm wheezing accessory muscles"),
    "J18" : ("Pneumonia J18: fever productive cough pleuritic chest pain dyspnea "
             "lobar consolidation infiltrate leukocytosis elevated CRP antibiotics "
             "community-acquired hospital-acquired aspiration sputum cultures"),
    "J44" : ("COPD J44: chronic airflow obstruction emphysema chronic bronchitis "
             "dyspnea exacerbation FEV1 bronchodilator steroids oxygen home oxygen"),
    "J45" : ("Asthma J45: episodic wheezing bronchospasm reversible airflow obstruction "
             "inhaler albuterol corticosteroids peak flow atopy allergic"),
    "J96" : ("Respiratory failure J96: hypoxia hypercapnia PaO2 PaCO2 intubation "
             "mechanical ventilation oxygen supplementation BiPAP CPAP ABG pH"),
    "J13" : ("Streptococcal pneumonia J13: Streptococcus pneumoniae lobar pneumonia "
             "fever consolidation bacteremia blood cultures penicillin"),
    "J20" : ("Acute bronchitis J20: cough sputum viral respiratory infection "
             "self-limiting bronchodilator inhaler"),
    # ── Cardiac / Cardiovascular ─────────────────────────────────────────────
    "I"   : ("Cardiovascular: chest pain dyspnea edema palpitations syncope "
             "cardiac markers troponin BNP EKG echocardiogram"),
    "I50" : ("Heart failure I50: dyspnea orthopnea paroxysmal nocturnal dyspnea "
             "peripheral edema elevated BNP reduced EF pulmonary congestion "
             "diuretics ACE inhibitor beta-blocker cardiomegaly JVD"),
    "I25" : ("Ischemic heart disease I25: stable angina exertional chest pressure "
             "troponin EKG ST changes coronary artery disease catheterization stent"),
    "I21" : ("Acute MI STEMI NSTEMI I21: acute chest pain crushing pressure radiation "
             "arm jaw troponin elevation CKMB ST segment elevation thrombolysis PCI "
             "aspirin heparin nitrates fibrinolysis"),
    "I48" : ("Atrial fibrillation I48: irregular rhythm palpitations rapid ventricular "
             "rate stroke risk anticoagulation warfarin DOAC rate control beta-blocker "
             "cardioversion ablation CHA2DS2-VASc"),
    "I10" : ("Hypertension I10: elevated blood pressure end-organ damage kidney "
             "hypertensive urgency emergency headache ACE inhibitor ARB diuretic"),
    "I63" : ("Ischemic stroke I63: sudden focal neurological deficit hemiplegia "
             "aphasia CT brain MRI DWI tPA thrombolysis antiplatelet anticoagulation"),
    "I47" : ("Supraventricular tachycardia I47: rapid heart rate palpitations "
             "adenosine vagal maneuver cardioversion EPS ablation"),
    # ── Infection / Sepsis ────────────────────────────────────────────────────
    "A"   : ("Infectious disease: fever leukocytosis elevated WBC cultures antibiotics "
             "source control bacteremia viremia fungemia immunocompromised"),
    "A41" : ("Sepsis septic shock A41: organ dysfunction hypotension tachycardia "
             "elevated lactate >2 mmol positive blood cultures broad-spectrum antibiotics "
             "fluid resuscitation vasopressors norepinephrine SOFA qSOFA ICU"),
    "A40" : ("Streptococcal sepsis A40: group A streptococcus bacteremia penicillin "
             "source control necrotizing fasciitis"),
    "B37" : ("Candidiasis B37: fungal infection immunocompromised central line "
             "azole fluconazole echinocandin blood cultures"),
    # ── Renal ─────────────────────────────────────────────────────────────────
    "N"   : ("Renal: creatinine BUN GFR urine output proteinuria electrolytes "
             "urinalysis renal function acute chronic kidney"),
    "N17" : ("Acute kidney injury AKI N17: rapid creatinine rise BUN elevated "
             "oliguria anuria prerenal intrinsic postrenal IV fluids dialysis "
             "nephrotoxin avoidance contrast creatinine AKIN KDIGO stages"),
    "N18" : ("Chronic kidney disease CKD N18: GFR <60 proteinuria anemia "
             "hypertension phosphate calcium parathyroid erythropoietin dialysis "
             "transplant uremia"),
    "N39" : ("Urinary tract infection N39: dysuria frequency urgency pyuria "
             "positive urine culture nitrites leukocyte esterase antibiotics "
             "complicated uncomplicated pyelonephritis"),
    # ── Metabolic / Endocrine ─────────────────────────────────────────────────
    "E"   : ("Metabolic endocrine: glucose electrolytes sodium potassium "
             "thyroid hormone insulin HbA1c metabolic panel"),
    "E11" : ("Type 2 diabetes mellitus E11: hyperglycemia HbA1c >6.5% "
             "insulin resistance polyuria polydipsia metformin insulin GLP-1 "
             "SGLT2 DKA HHS complications neuropathy nephropathy retinopathy"),
    "E10" : ("Type 1 diabetes E10: autoimmune insulin-dependent DKA ketones "
             "insulin pump basal bolus glucose monitoring"),
    "E87" : ("Electrolyte disorder E87: hyponatremia hypernatremia hypokalemia "
             "hyperkalemia hypomagnesemia sodium correction fluid management IV"),
    "E86" : ("Volume depletion dehydration E86: hypovolemia orthostatic hypotension "
             "tachycardia dry mucous membranes decreased urine output IV fluids"),
    "E66" : ("Obesity E66: BMI >30 metabolic syndrome insulin resistance sleep apnea"),
    # ── GI / Hepatic ──────────────────────────────────────────────────────────
    "K"   : ("Gastrointestinal: abdominal pain nausea vomiting diarrhea LFTs "
             "amylase lipase imaging ultrasound CT scan"),
    "K80" : ("Cholelithiasis cholecystitis K80: right upper quadrant pain Murphy sign "
             "ultrasound gallstones bile duct cholecystectomy ERCP"),
    "K92" : ("GI hemorrhage K92: hematemesis melena hematochezia hemoglobin drop "
             "endoscopy colonoscopy IV PPI blood transfusion"),
    "K85" : ("Acute pancreatitis K85: epigastric pain elevated amylase lipase "
             "nausea vomiting CT scan severity gallstone alcohol Ranson Balthazar"),
    "K21" : ("GERD K21: heartburn acid reflux esophagitis PPI antacid Barrets"),
    "K57" : ("Diverticular disease K57: left lower quadrant pain fever diverticulitis "
             "CT scan antibiotics perforation"),
    # ── Mental Health ──────────────────────────────────────────────────────────
    "F"   : ("Psychiatric mental health: mood affect cognition behavior orientation "
             "psychiatric assessment PHQ GAD"),
    "F32" : ("Major depressive disorder F32: low mood anhedonia sleep disturbance "
             "fatigue concentration poor suicidal ideation antidepressant SSRI SNRI"),
    "F41" : ("Anxiety disorder F41: excessive worry panic attack autonomic symptoms "
             "tachycardia diaphoresis benzodiazepine SSRI"),
    "F10" : ("Alcohol use disorder F10: withdrawal tremor seizure delirium thiamine "
             "CIWA benzodiazepine detox"),
    # ── Injury / Trauma ────────────────────────────────────────────────────────
    "S"   : "Trauma injury: mechanism imaging stabilisation hemorrhage orthopedic fracture",
    "T"   : "Poisoning overdose: toxicology antidote supportive care gastric lavage activated charcoal",
    # ── Neoplasm ──────────────────────────────────────────────────────────────
    "C"   : ("Malignancy cancer C: histology pathology staging chemotherapy radiation "
             "surgery biopsy metastasis lymph node tumor marker"),
    "C34" : ("Lung cancer C34: cough hemoptysis weight loss CT scan PET biopsy "
             "NSCLC SCLC staging chemotherapy immunotherapy"),
    "C50" : ("Breast cancer C50: mass lumpectomy mastectomy hormone receptor HER2 "
             "chemotherapy radiation tamoxifen"),
    "D"   : "Benign neoplasm D: imaging biopsy excision surveillance",
    "D50" : ("Iron deficiency anemia D50: microcytic hypochromic low ferritin "
             "iron supplementation GI blood loss"),
    "D64" : ("Anemia D64: hemoglobin CBC iron B12 folate reticulocyte transfusion "
             "erythropoietin chronic disease hemolysis"),
    # ── Neurological ──────────────────────────────────────────────────────────
    "G"   : ("Neurological: mental status level of consciousness focal deficits "
             "imaging CT MRI EEG"),
    "G20" : ("Parkinson disease G20: tremor rigidity bradykinesia dopamine levodopa "
             "carbidopa DBS gait freezing"),
    "G40" : ("Epilepsy G40: seizure EEG anticonvulsant levetiracetam phenytoin valproate "
             "status epilepticus benzodiazepine lorazepam"),
    "G89" : ("Pain G89: acute chronic assessment multimodal analgesia opioid NSAID "
             "neuropathic gabapentin pregabalin"),
    # ── Musculoskeletal ───────────────────────────────────────────────────────
    "M"   : ("Musculoskeletal: joint pain swelling arthritis gout uric acid "
             "rheumatoid autoimmune NSAID corticosteroid"),
    "M79" : ("Soft tissue disorder M79: myalgia fibromyalgia pain physical therapy"),
    # ── Blood / Coagulation ───────────────────────────────────────────────────
    "D68" : ("Coagulopathy D68: prolonged PT INR PTT bleeding anticoagulation reversal "
             "FFP vitamin K factor deficiency DIC"),
    "D69" : ("Thrombocytopenia D69: low platelet count bleeding hematology ITP HIT TTP"),
    # ── Genitourinary ──────────────────────────────────────────────────────────
    "N20" : ("Nephrolithiasis kidney stone N20: flank pain hematuria CT scan "
             "stone size urology lithotripsy"),
    "N40" : ("Benign prostatic hyperplasia N40: urinary retention hesitancy frequency "
             "PSA urinalysis alpha-blocker 5-alpha reductase"),
}


def build_evidence_corpus(
    top50_codes: List[str],
    df_train: pd.DataFrame,
    prototypes_per_dx: int = config.PROTOTYPES_PER_DX,
    seed: int = config.SEED,
) -> List[dict]:
    """
    Build the evidence corpus for Phase 3 RAG.

    Two sources:
      1. Clinical knowledge base  — one passage per ICD chapter / code match.
      2. MIMIC prototypes         — up to *prototypes_per_dx* real notes
                                    per diagnosis from the training set.

    Returns a list of dicts:  {"text": str, "diagnosis": str, "source": str}
    """
    corpus: List[dict] = []

    # --- Clinical knowledge --------------------------------------------------
    for code in top50_codes:
        matched = False
        # Try longest-prefix match first (specific → general)
        for prefix in sorted(_CLINICAL_KB.keys(), key=len, reverse=True):
            if code.startswith(prefix):
                text = _CLINICAL_KB[prefix]
                corpus.append({
                    "text"     : f"{code}: {text}",
                    "diagnosis": code,
                    "source"   : "clinical_knowledge",
                })
                matched = True
                break
        if not matched:
            corpus.append({
                "text"     : f"{code}: Clinical diagnosis requiring diagnostic correlation "
                             "and workup with relevant labs imaging and clinical history",
                "diagnosis": code,
                "source"   : "clinical_knowledge",
            })

    n_kb = len([c for c in corpus if c["source"] == "clinical_knowledge"])
    log.info(f"Clinical knowledge passages: {n_kb}")

    # --- MIMIC prototypes ----------------------------------------------------
    code_idx_map = {code: i for i, code in enumerate(top50_codes)}
    for dx_code in tqdm(top50_codes, desc="Sampling MIMIC prototypes"):
        idx = code_idx_map[dx_code]

        # Support both column-based and list-of-labels formats
        if dx_code in df_train.columns:
            positives = df_train[df_train[dx_code] == 1]
        elif "labels" in df_train.columns:
            positives = df_train[df_train["labels"].apply(
                lambda x: isinstance(x, list) and len(x) > idx and x[idx] == 1
            )]
        else:
            positives = pd.DataFrame()

        n = min(len(positives), prototypes_per_dx)
        if n > 0:
            for _, row in positives.sample(n=n, random_state=seed).iterrows():
                text = str(row["text"])
                # Keep first 400 chars — captures chief complaint and key findings
                corpus.append({
                    "text"     : text[:400],
                    "diagnosis": dx_code,
                    "source"   : "mimic_prototype",
                })

    n_mimic = len([c for c in corpus if c["source"] == "mimic_prototype"])
    log.info(
        f"Evidence corpus: {len(corpus)} passages  "
        f"({n_kb} clinical KB + {n_mimic} MIMIC prototypes, "
        f"{prototypes_per_dx} per dx)"
    )
    return corpus


# ============================================================================
# SIMPLE RAG (FAISS)
# ============================================================================

class SimpleRAG:
    """
    FAISS-backed retriever.

    Usage:
        rag = SimpleRAG()
        rag.build_index(corpus, index_cache_path=config.FAISS_INDEX)
        text = rag.retrieve("sepsis hypotension lactate fever")

    The index is CPU-only — MPS has no GPU FAISS support.

    Notes on query strategy:
        Phase 3 passes concept-name strings (e.g. "sepsis pneumonia tachycardia")
        rather than full clinical notes.  Short keyword queries achieve much
        higher cosine similarity with the KB passages, which use the same
        clinical vocabulary.
    """

    def __init__(
        self,
        model_name: str = config.RAG_MODEL_NAME,
        top_k: int = config.RAG_TOP_K,
        threshold: float = config.RAG_THRESHOLD,
    ) -> None:
        log.info(f"Loading RAG encoder: {model_name}")
        # Sentence-transformers works fine on CPU; MPS support is partial
        self.encoder   = SentenceTransformer(model_name, device="cpu")
        self.top_k     = top_k
        self.threshold = threshold
        self.index: Optional[faiss.Index] = None
        self.documents: List[dict] = []

    # ------------------------------------------------------------------
    def build_index(
        self,
        documents: List[dict],
        index_cache_path: Optional[Path] = None,
    ) -> None:
        """
        Build (or load from cache) the FAISS index.

        Args:
            documents        : list of {"text": str, …}
            index_cache_path : path to save / load the binary index.
                               Cache is invalidated when corpus size changes.
        """
        self.documents = documents
        cache = Path(index_cache_path) if index_cache_path else None

        if cache and cache.exists():
            loaded_index = faiss.read_index(str(cache))
            # Invalidate cache if corpus grew (new prototypes_per_dx etc.)
            if loaded_index.ntotal == len(documents):
                self.index = loaded_index
                log.info(
                    f"FAISS index loaded from cache "
                    f"({self.index.ntotal} vectors)  ← {cache.name}"
                )
                return
            else:
                log.warning(
                    f"FAISS cache size ({loaded_index.ntotal}) != corpus size "
                    f"({len(documents)}) — rebuilding index."
                )

        log.info(f"Building FAISS index from {len(documents)} passages …")
        texts = [d["text"] for d in documents]

        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=config.RAG_ENCODE_BATCH_SIZE,
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
        self.index.add(embeddings)

        log.info(f"FAISS index built: {embeddings.shape[1]}-dim, {self.index.ntotal} vectors")

        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(cache))
            log.info(f"FAISS index saved → {cache.name}")

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> str:
        """
        Retrieve the top-K most relevant passages and return them as a
        single concatenated string (empty string if none exceed threshold).
        """
        if self.index is None or not query.strip():
            return ""

        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(self.top_k, self.index.ntotal))

        texts = [
            self.documents[int(idx)]["text"]
            for score, idx in zip(scores[0], indices[0])
            if score >= self.threshold and idx >= 0
        ]
        return " | ".join(texts)

    # ------------------------------------------------------------------
    def retrieve_batch(self, queries: List[str]) -> List[str]:
        """
        Encode all queries in one call and return retrieved passage strings.
        More efficient than calling retrieve() in a Python loop when batch > 1.
        """
        if self.index is None or not queries:
            return [""] * len(queries)

        # Encode all queries at once
        q_embs = self.encoder.encode(
            [q if q.strip() else "N/A" for q in queries],
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=len(queries),
        ).astype("float32")
        faiss.normalize_L2(q_embs)

        scores_all, indices_all = self.index.search(
            q_embs, min(self.top_k, self.index.ntotal)
        )

        results = []
        for q, scores_row, indices_row in zip(queries, scores_all, indices_all):
            if not q.strip():
                results.append("")
                continue
            texts = [
                self.documents[int(idx)]["text"]
                for score, idx in zip(scores_row, indices_row)
                if score >= self.threshold and idx >= 0
            ]
            results.append(" | ".join(texts))
        return results
