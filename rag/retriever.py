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
    "J"   : ("Respiratory disease: cough dyspnea hypoxia tachypnea respiratory failure "
             "oxygen saturation SpO2 pulmonary infiltrate chest X-ray bronchospasm "
             "wheezing accessory muscle use retractions pleural effusion atelectasis "
             "ventilation perfusion mismatch intubation"),
    "J18" : ("Pneumonia J18: fever chills productive cough pleuritic chest pain dyspnea "
             "tachycardia tachypnea hypoxia lobar consolidation infiltrate leukocytosis "
             "elevated CRP procalcitonin community-acquired hospital-acquired aspiration "
             "sputum cultures blood cultures antibiotics azithromycin ceftriaxone "
             "beta-lactam respiratory distress bronchopneumonia interstitial pneumonia "
             "viral bacterial atypical Legionella Mycoplasma"),
    "J44" : ("COPD chronic obstructive pulmonary disease J44: chronic airflow obstruction "
             "emphysema chronic bronchitis dyspnea exacerbation FEV1 FVC ratio <0.70 "
             "bronchodilator albuterol ipratropium tiotropium steroids prednisone "
             "oxygen home oxygen LTOT hypercapnia CO2 retention barrel chest pursed lip "
             "pink puffer blue bloater cor pulmonale pulmonary hypertension"),
    "J45" : ("Asthma J45: episodic wheezing bronchospasm reversible airflow obstruction "
             "dyspnea chest tightness cough inhaler albuterol beta-agonist "
             "corticosteroids budesonide fluticasone peak flow meter atopy allergic "
             "triggers exercise cold air IgE eosinophils spirometry steroid burst "
             "status asthmaticus nebulizer magnesium"),
    "J96" : ("Respiratory failure J96: hypoxia hypercapnia PaO2 <60 mmHg PaCO2 >50 "
             "type 1 type 2 failure intubation mechanical ventilation BiPAP CPAP "
             "oxygen supplementation ABG arterial blood gas pH acidosis ARDS "
             "acute respiratory distress syndrome weaning extubation"),
    "J13" : ("Streptococcal pneumonia J13: Streptococcus pneumoniae lobar pneumonia "
             "fever rigors productive cough rust-coloured sputum consolidation "
             "bacteremia blood cultures penicillin amoxicillin"),
    "J20" : ("Acute bronchitis J20: cough purulent sputum viral respiratory tract infection "
             "self-limiting bronchodilator fever malaise URI upper respiratory infection"),
    # ── Cardiac / Cardiovascular ─────────────────────────────────────────────
    "I"   : ("Cardiovascular disease: chest pain dyspnea edema palpitations syncope "
             "presyncope cardiac markers troponin BNP NT-proBNP EKG ECG echocardiogram "
             "tachycardia bradycardia arrhythmia heart murmur JVD jugular venous distension"),
    "I50" : ("Heart failure congestive cardiac failure I50: dyspnea orthopnea paroxysmal "
             "nocturnal dyspnea PND exertional dyspnea peripheral edema leg swelling "
             "elevated BNP NT-proBNP reduced ejection fraction EF systolic diastolic "
             "pulmonary congestion crackles S3 gallop cardiomegaly diuretics furosemide "
             "ACE inhibitor ARB beta-blocker cardiomyopathy fluid overload weight gain "
             "bilateral crackles pleural effusion JVD"),
    "I25" : ("Ischemic heart disease coronary artery disease I25: stable angina "
             "exertional chest pressure tightness radiation arm jaw troponin EKG ST "
             "changes T-wave inversion stress test catheterization PCI stent CABG "
             "antiplatelet aspirin clopidogrel statin atherosclerosis risk factors"),
    "I21" : ("Acute myocardial infarction MI STEMI NSTEMI I21: acute chest pain "
             "crushing pressure radiation left arm jaw diaphoresis nausea troponin "
             "elevation CKMB CK ST segment elevation thrombolysis PCI angioplasty "
             "aspirin heparin nitrates fibrinolysis door-to-balloon time reperfusion "
             "cardiogenic shock arrhythmia ventricular fibrillation"),
    "I48" : ("Atrial fibrillation AFib I48: irregular heart rhythm palpitations "
             "rapid ventricular rate stroke risk anticoagulation warfarin DOAC "
             "apixaban rivaroxaban rate control beta-blocker diltiazem digoxin "
             "cardioversion ablation CHA2DS2-VASc HAS-BLED thyrotoxicosis valvular "
             "lone paroxysmal persistent permanent flutter"),
    "I10" : ("Hypertension HTN I10: elevated blood pressure systolic diastolic "
             "end-organ damage hypertensive urgency emergency hypertensive crisis "
             "headache visual changes kidney damage proteinuria ACE inhibitor ARB "
             "calcium channel blocker diuretic thiazide amlodipine lisinopril"),
    "I63" : ("Ischemic stroke cerebrovascular accident CVA I63: sudden focal "
             "neurological deficit hemiplegia hemiparesis aphasia dysarthria "
             "facial droop arm drift CT brain MRI diffusion DWI tPA alteplase "
             "thrombolysis thrombectomy antiplatelet anticoagulation NIHSS "
             "atrial fibrillation carotid stenosis"),
    "I47" : ("Supraventricular tachycardia SVT I47: rapid heart rate palpitations "
             "dizziness presyncope narrow complex tachycardia adenosine vagal maneuver "
             "Valsalva cardioversion EPS electrophysiology ablation"),
    # ── Infection / Sepsis ────────────────────────────────────────────────────
    "A"   : ("Infectious disease infection: fever chills rigors leukocytosis "
             "elevated WBC neutrophilia cultures blood urine sputum antibiotics "
             "source control bacteremia viremia fungemia immunocompromised "
             "temperature >38.3 or <36 tachycardia tachypnea"),
    "A41" : ("Sepsis septic shock systemic inflammatory response A41: organ dysfunction "
             "hypotension tachycardia tachypnea fever chills altered mental status "
             "elevated lactate >2 mmol/L positive blood cultures broad-spectrum "
             "antibiotics piperacillin-tazobactam vancomycin fluid resuscitation "
             "crystalloid vasopressors norepinephrine dopamine SOFA qSOFA ICU "
             "septic shock MAP <65 despite fluid oliguria creatinine rise bilirubin"),
    "A40" : ("Streptococcal sepsis A40: group A streptococcus GAS bacteremia "
             "penicillin source control necrotizing fasciitis cellulitis wound infection"),
    "B37" : ("Candidiasis fungal infection B37: Candida immunocompromised neutropenic "
             "central line catheter azole fluconazole echinocandin caspofungin "
             "blood cultures oral thrush esophageal candidemia"),
    # ── Renal ─────────────────────────────────────────────────────────────────
    "N"   : ("Renal kidney disease: creatinine BUN GFR urine output proteinuria "
             "electrolytes urinalysis renal function acute chronic kidney failure "
             "hematuria casts oliguria anuria edema hypertension"),
    "N17" : ("Acute kidney injury AKI N17: rapid creatinine rise BUN elevated "
             "oliguria anuria urine output <0.5 ml/kg/hr prerenal intrinsic postrenal "
             "obstruction IV fluids dialysis CRRT nephrotoxin avoidance contrast "
             "NSAID aminoglycoside AKIN KDIGO stages tubular necrosis ATN ischemia "
             "renal hypoperfusion shock sepsis"),
    "N18" : ("Chronic kidney disease CKD N18: GFR <60 ml/min/1.73m2 proteinuria "
             "anemia normocytic hypertension phosphate calcium parathyroid PTH "
             "erythropoietin ESA dialysis hemodialysis peritoneal transplant uremia "
             "hyperkalemia acidosis fluid retention edema"),
    "N39" : ("Urinary tract infection UTI N39: dysuria frequency urgency pyuria "
             "bacteriuria positive urine culture nitrites leukocyte esterase WBC "
             "antibiotics trimethoprim nitrofurantoin ciprofloxacin complicated "
             "uncomplicated pyelonephritis fever flank pain costovertebral tenderness"),
    # ── Metabolic / Endocrine ─────────────────────────────────────────────────
    "E"   : ("Metabolic endocrine disorder: glucose hyperglycemia hypoglycemia "
             "electrolytes sodium potassium thyroid hormone insulin HbA1c hemoglobin A1c "
             "metabolic panel basic comprehensive BMP CMP"),
    "E11" : ("Type 2 diabetes mellitus T2DM E11: hyperglycemia elevated HbA1c >6.5% "
             "insulin resistance polyuria polydipsia polyphagia weight loss fatigue "
             "metformin insulin GLP-1 agonist SGLT2 inhibitor DKA diabetic ketoacidosis "
             "HHS hyperosmolar hyperglycemic state microvascular macrovascular "
             "neuropathy peripheral nephropathy retinopathy cardiovascular risk "
             "glucose monitoring fingerstick"),
    "E10" : ("Type 1 diabetes mellitus T1DM E10: autoimmune insulin-dependent "
             "DKA ketoacidosis anion gap metabolic acidosis ketones ketonuria "
             "insulin pump continuous subcutaneous basal bolus blood glucose monitoring "
             "hypoglycemia glucagon C-peptide absent antibodies GAD"),
    "E87" : ("Electrolyte disorder E87: hyponatremia sodium <135 hypernatremia "
             "sodium >145 hypokalemia potassium <3.5 hyperkalemia >5.5 EKG changes "
             "peaked T waves hypomagnesemia hypocalcemia sodium correction rate "
             "SIADH cerebral salt wasting fluid management IV replacement"),
    "E86" : ("Volume depletion dehydration hypovolemia E86: orthostatic hypotension "
             "tachycardia dry mucous membranes skin turgor decreased urine output "
             "concentrated urine elevated BUN creatinine ratio IV fluids normal saline "
             "lactated Ringer's poor oral intake vomiting diarrhea fever"),
    "E66" : ("Obesity E66: BMI >30 body mass index metabolic syndrome insulin "
             "resistance dyslipidemia hypertension obstructive sleep apnea OSA "
             "NAFLD non-alcoholic fatty liver bariatric surgery weight management"),
    # ── GI / Hepatic ──────────────────────────────────────────────────────────
    "K"   : ("Gastrointestinal GI disorder: abdominal pain nausea vomiting diarrhea "
             "LFTs liver function tests amylase lipase imaging ultrasound CT abdomen "
             "bowel obstruction distension tenderness guarding rebound"),
    "K80" : ("Cholelithiasis cholecystitis biliary colic K80: right upper quadrant "
             "pain Murphy sign positive fever nausea vomiting ultrasound gallstones "
             "bile duct common bile duct dilation cholecystectomy ERCP laparoscopic "
             "acute chronic acalculous elevated alkaline phosphatase bilirubin"),
    "K92" : ("GI hemorrhage gastrointestinal bleeding K92: hematemesis coffee-ground "
             "emesis melena black tarry stool hematochezia hemoglobin drop transfusion "
             "endoscopy upper lower colonoscopy IV PPI proton pump inhibitor "
             "peptic ulcer variceal portal hypertension diverticular angiodysplasia"),
    "K85" : ("Acute pancreatitis K85: severe epigastric pain radiation to back "
             "elevated amylase lipase nausea vomiting CT scan Balthazar severity "
             "gallstone biliary alcohol Ranson criteria APACHE necrotizing hemorrhagic "
             "pseudocyst pleural effusion fluid resuscitation NPO bowel rest"),
    "K21" : ("GERD gastroesophageal reflux disease K21: heartburn pyrosis acid reflux "
             "regurgitation esophagitis erosive PPI proton pump inhibitor antacid "
             "Barrett's esophagus dysplasia hiatal hernia dysphagia"),
    "K57" : ("Diverticular disease diverticulitis K57: left lower quadrant LLQ pain "
             "fever leukocytosis CT scan antibiotics perforation abscess peritonitis "
             "colonic diverticula microperforation bowel rest"),
    # ── Mental Health ──────────────────────────────────────────────────────────
    "F"   : ("Psychiatric mental health disorder: mood affect cognition behavior "
             "orientation psychiatric assessment PHQ-9 GAD-7 MMSE cognition "
             "psychosis hallucinations delusions agitation"),
    "F32" : ("Major depressive disorder MDD depression F32: persistent low mood "
             "anhedonia sleep disturbance insomnia hypersomnia fatigue poor concentration "
             "worthlessness hopelessness suicidal ideation psychomotor retardation "
             "antidepressant SSRI SNRI venlafaxine sertraline fluoxetine CBT"),
    "F41" : ("Anxiety disorder F41: excessive worry generalized panic attack "
             "autonomic symptoms tachycardia diaphoresis palpitations chest tightness "
             "tremor shortness of breath derealization benzodiazepine lorazepam "
             "clonazepam SSRI buspirone CBT"),
    "F10" : ("Alcohol use disorder alcoholism F10: alcohol dependence withdrawal "
             "tremor seizure delirium tremens DTs diaphoresis agitation thiamine "
             "Wernicke Korsakoff CIWA scale benzodiazepine chlordiazepoxide detox "
             "elevated GGT AST:ALT ratio"),
    # ── Injury / Trauma ────────────────────────────────────────────────────────
    "S"   : ("Trauma injury S: mechanism blunt penetrating imaging fracture dislocation "
             "hemorrhage hemorrhagic shock stabilisation ATLS primary survey airway "
             "breathing circulation orthopedic surgery internal fixation ORIF"),
    "T"   : ("Poisoning overdose toxicity T: toxicology antidote supportive care "
             "gastric lavage activated charcoal acetylcysteine naloxone flumazenil "
             "drug level serum acetaminophen salicylate opioid benzodiazepine"),
    # ── Neoplasm ──────────────────────────────────────────────────────────────
    "C"   : ("Malignancy cancer neoplasm C: histology pathology staging TNM "
             "chemotherapy radiation surgery biopsy metastasis lymph node tumor "
             "marker CA-125 CEA PSA weight loss fatigue cachexia"),
    "C34" : ("Lung cancer carcinoma C34: cough hemoptysis weight loss dyspnea "
             "CT chest PET scan biopsy bronchoscopy NSCLC SCLC adenocarcinoma "
             "squamous cell staging chemotherapy platinum immunotherapy pembrolizumab "
             "EGFR ALK mutation targeted therapy"),
    "C50" : ("Breast cancer carcinoma C50: breast mass lump lumpectomy mastectomy "
             "hormone receptor ER PR HER2 positive negative sentinel node biopsy "
             "axillary dissection chemotherapy radiation tamoxifen aromatase inhibitor "
             "trastuzumab adjuvant neoadjuvant"),
    "D"   : ("Benign neoplasm D: imaging ultrasound CT MRI biopsy excision surveillance "
             "watchful waiting incidental finding polyp lipoma cyst"),
    "D50" : ("Iron deficiency anemia IDA D50: microcytic hypochromic low MCV "
             "low ferritin iron deficiency GI blood loss menorrhagia malabsorption "
             "iron supplementation IV iron blood loss workup colonoscopy"),
    "D64" : ("Anemia D64: low hemoglobin hematocrit CBC iron studies B12 folate "
             "reticulocyte count transfusion erythropoietin chronic disease "
             "hemolysis hemolytic direct Coombs aplastic pernicious macrocytic"),
    # ── Neurological ──────────────────────────────────────────────────────────
    "G"   : ("Neurological disease: mental status altered consciousness focal deficits "
             "weakness sensory loss coordination gait imaging CT MRI EEG lumbar puncture "
             "CSF headache seizure syncope dementia encephalopathy"),
    "G20" : ("Parkinson disease PD G20: resting tremor rigidity cogwheel bradykinesia "
             "shuffling gait festination postural instability dopamine dopaminergic "
             "levodopa carbidopa pramipexole DBS deep brain stimulation micrographia "
             "masked facies freezing falls substantia nigra"),
    "G40" : ("Epilepsy seizure disorder G40: tonic clonic grand mal focal partial "
             "EEG electroencephalogram anticonvulsant antiepileptic levetiracetam "
             "phenytoin valproate carbamazepine lamotrigine status epilepticus "
             "benzodiazepine lorazepam diazepam postictal aura"),
    "G89" : ("Pain management G89: acute chronic pain numeric rating scale NRS "
             "multimodal analgesia opioid morphine hydromorphone oxycodone fentanyl "
             "NSAID ibuprofen ketorolac neuropathic gabapentin pregabalin duloxetine "
             "pain assessment sedation analgesic ladder"),
    # ── Musculoskeletal ───────────────────────────────────────────────────────
    "M"   : ("Musculoskeletal rheumatologic disease: joint pain arthralgia swelling "
             "arthritis gout uric acid hyperuricemia inflammatory rheumatoid "
             "autoimmune ANA RF CCP NSAID corticosteroid DMARDs"),
    "M79" : ("Soft tissue musculoskeletal disorder M79: myalgia muscle pain "
             "fibromyalgia widespread pain fatigue tender points physical therapy "
             "trigger point injection NSAID low-impact exercise"),
    # ── Blood / Coagulation ───────────────────────────────────────────────────
    "D68" : ("Coagulopathy bleeding disorder D68: prolonged PT INR PTT elevated "
             "bleeding bruising anticoagulation reversal FFP fresh frozen plasma "
             "vitamin K factor deficiency DIC disseminated intravascular coagulation "
             "hemophilia factor VIII IX warfarin reversal platelet transfusion"),
    "D69" : ("Thrombocytopenia low platelet count D69: bleeding petechiae purpura "
             "ecchymosis spontaneous bleeding hematology ITP immune thrombocytopenic "
             "HIT heparin-induced TTP thrombotic thrombocytopenic purpura ADAMTS13 "
             "platelet transfusion threshold <10,000 <50,000"),
    # ── Genitourinary ──────────────────────────────────────────────────────────
    "N20" : ("Nephrolithiasis kidney stone urolithiasis N20: acute severe flank pain "
             "colicky radiation groin hematuria gross microscopic CT KUB non-contrast "
             "stone size hydronephrosis urology lithotripsy ESWL ureteroscopy"),
    "N40" : ("Benign prostatic hyperplasia BPH N40: lower urinary tract symptoms LUTS "
             "urinary retention hesitancy weak stream frequency nocturia overflow "
             "incontinence PSA prostate specific antigen urinalysis post-void residual "
             "alpha-blocker tamsulosin 5-alpha reductase finasteride"),
    # ── Pulmonary vascular ────────────────────────────────────────────────────
    "I26" : ("Pulmonary embolism PE I26: sudden dyspnea pleuritic chest pain "
             "tachycardia tachypnea hemoptysis hypoxia Wells score D-dimer "
             "CT pulmonary angiogram CTPA VQ scan anticoagulation heparin "
             "LMWH enoxaparin warfarin DOAC massive submassive saddle PE "
             "right heart strain troponin BNP thrombolysis"),
    "I82" : ("Deep vein thrombosis DVT I82: unilateral leg swelling pain erythema "
             "warmth Homans sign Wells score D-dimer duplex ultrasound venous "
             "Doppler anticoagulation LMWH heparin thrombosis thrombus occlusion"),
}


def build_evidence_corpus(
    top50_codes: List[str],
    df_train: pd.DataFrame,
    prototypes_per_dx: int = config.PROTOTYPES_PER_DX,
    seed: int = config.SEED,
    pubmed_path: Optional[Path] = None,
) -> List[dict]:
    """
    Build the evidence corpus for Phase 3 RAG.

    Three sources:
      1. Clinical knowledge base  — one passage per ICD chapter / code match.
      2. MIMIC prototypes         — up to *prototypes_per_dx* real notes
                                    per diagnosis from the training set.
      3. PubMed abstracts (optional) — genuine clinical literature downloaded
                                    via scripts/download_pubmed.py.  This is
                                    the only source of truly new knowledge not
                                    already in the model weights.

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

    # --- PubMed abstracts (new clinical knowledge, not in model weights) ------
    pubmed_file = pubmed_path or (config.EVIDENCE_P3 / "pubmed_abstracts.json")
    n_pubmed = 0
    if pubmed_file.exists():
        with open(pubmed_file) as f:
            pubmed_docs = json.load(f)
        for doc in pubmed_docs:
            # Only include if the diagnosis code is in our top-50 set
            diag = doc.get("diagnosis", "")
            if diag in top50_codes or any(diag.startswith(c) or c.startswith(diag) for c in top50_codes):
                corpus.append({
                    "text"     : str(doc.get("text", ""))[:600],
                    "diagnosis": diag,
                    "source"   : "pubmed",
                })
                n_pubmed += 1
        log.info(f"PubMed abstracts added: {n_pubmed}  ← {pubmed_file.name}")
    else:
        log.info(
            f"No PubMed abstracts found at {pubmed_file.name}. "
            "Run scripts/download_pubmed.py to add clinical literature."
        )

    n_mimic = len([c for c in corpus if c["source"] == "mimic_prototype"])
    log.info(
        f"Evidence corpus: {len(corpus)} passages  "
        f"({n_kb} clinical KB + {n_mimic} MIMIC prototypes + {n_pubmed} PubMed)"
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
