#!/usr/bin/env python3
"""
scripts/download_pubmed.py

Downloads PubMed abstracts for the Top-50 ICD-10 diagnosis codes and
saves them as a JSON file that the Phase 3 evidence corpus builder can
merge with the MIMIC prototypes.

This gives RAG access to genuine clinical literature — new information that
the model did NOT see during training.  MIMIC prototypes alone are not enough
because the model already memorised the training set.

Run (once, before phase3_train.py):
    cd ShifaMind_Local
    python scripts/download_pubmed.py

Reads:
    shifamind_local/shared_data/top50_icd10_info.json

Writes:
    shifamind_local/evidence_store/pubmed_abstracts.json

Requirements:
    biopython (already in requirements.txt)
    No API key needed — NCBI Entrez allows ~3 requests/second unauthenticated.
    Set NCBI_EMAIL in .env to increase to 10 requests/second.

Usage tips:
  • First run downloads from NCBI (~2-5 min depending on network).
  • Subsequent runs load from cache (no re-download).
  • Use --rebuild to force fresh download.
  • Use --max-per-code N to limit abstracts per code (default 30).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.logging_utils import get_logger

log = get_logger()

# ── ICD-10 code → MeSH / search query mappings ──────────────────────────────
# PubMed doesn't index by ICD-10 directly; we use clinical search terms.
# Keys are ICD-10 codes (or prefixes), values are PubMed query strings.

_CODE_TO_QUERY = {
    # Cardiac
    "I10"  : "hypertension essential diagnosis clinical criteria",
    "I21"  : "acute myocardial infarction STEMI NSTEMI diagnosis",
    "I25"  : "chronic ischemic heart disease coronary artery disease",
    "I48"  : "atrial fibrillation diagnosis management anticoagulation",
    "I50"  : "heart failure diagnosis BNP ejection fraction treatment",
    "I63"  : "ischemic stroke acute management tPA thrombolysis",
    "I47"  : "supraventricular tachycardia SVT diagnosis treatment",
    # Respiratory
    "J18"  : "pneumonia community-acquired diagnosis treatment antibiotics",
    "J44"  : "COPD chronic obstructive pulmonary disease exacerbation",
    "J45"  : "asthma exacerbation management bronchospasm treatment",
    "J96"  : "respiratory failure acute hypoxemic hypercapnic intubation",
    "J13"  : "streptococcal pneumonia Streptococcus pneumoniae",
    "J20"  : "acute bronchitis viral respiratory infection management",
    # Sepsis / Infection
    "A41"  : "sepsis septic shock diagnosis organ dysfunction lactate",
    "A40"  : "streptococcal sepsis group A bacteremia",
    "B37"  : "candidiasis fungal infection invasive immunocompromised",
    # Renal
    "N17"  : "acute kidney injury AKI diagnosis KDIGO creatinine",
    "N18"  : "chronic kidney disease CKD GFR diagnosis management",
    "N39"  : "urinary tract infection UTI diagnosis treatment antibiotics",
    "N20"  : "nephrolithiasis kidney stone acute management ureter",
    "N40"  : "benign prostatic hyperplasia BPH urinary retention treatment",
    # Metabolic / Endocrine
    "E10"  : "type 1 diabetes mellitus DKA ketoacidosis management",
    "E11"  : "type 2 diabetes mellitus HbA1c hyperglycemia management",
    "E87"  : "electrolyte disorder hyponatremia hyperkalemia management",
    "E86"  : "dehydration volume depletion hypovolemia clinical assessment",
    "E66"  : "obesity metabolic syndrome clinical management bariatric",
    # GI / Hepatic
    "K80"  : "cholelithiasis cholecystitis diagnosis ultrasound cholecystectomy",
    "K85"  : "acute pancreatitis diagnosis severity CT scan management",
    "K92"  : "gastrointestinal bleeding upper lower endoscopy management",
    "K21"  : "gastroesophageal reflux GERD diagnosis PPI treatment",
    "K57"  : "diverticular disease diverticulitis CT management antibiotics",
    # Mental health
    "F10"  : "alcohol use disorder withdrawal CIWA management detoxification",
    "F32"  : "major depressive disorder diagnosis PHQ treatment antidepressant",
    "F41"  : "anxiety disorder panic attack diagnosis treatment SSRI",
    # Neurological
    "G40"  : "epilepsy seizure EEG anticonvulsant management status epilepticus",
    "G20"  : "Parkinson disease diagnosis dopamine treatment levodopa",
    "G89"  : "chronic pain management multimodal analgesia opioid neuropathic",
    # Blood / Coagulation
    "D50"  : "iron deficiency anemia diagnosis ferritin management",
    "D64"  : "anemia diagnosis CBC reticulocyte iron B12 transfusion",
    "D68"  : "coagulopathy DIC diagnosis anticoagulation reversal",
    "D69"  : "thrombocytopenia ITP HIT TTP platelet diagnosis management",
    # Musculoskeletal
    "M79"  : "myalgia fibromyalgia chronic pain diagnosis management",
    # Neoplasm
    "C34"  : "lung cancer NSCLC SCLC diagnosis staging treatment",
    "C50"  : "breast cancer diagnosis staging hormone receptor treatment",
    # Injury / External
    "S"    : "trauma injury mechanism stabilisation haemorrhage management",
    "T"    : "poisoning overdose toxicology antidote management",
    # Procedure / Status codes
    "Z951" : "coronary artery bypass history cardiac history",
    "Z955" : "coronary angioplasty stent history prior PCI",
    "Z23"  : "vaccination immunization preventive encounter",
    "Y92230": "healthcare facility encounter location",
    "Y92239": "hospital inpatient encounter setting",
    # Misc
    "E785" : "hyperlipidaemia dyslipidaemia statin LDL cholesterol management",
    "I2510": "atherosclerotic heart disease coronary artery disease native",
}


def _build_fallback_query(code: str) -> str:
    """Generate a fallback PubMed query from the ICD-10 code prefix."""
    prefixes = {
        "I": "cardiovascular disease cardiac management",
        "J": "respiratory disease pulmonary management",
        "A": "infectious disease sepsis bacteremia",
        "N": "renal disease kidney acute chronic",
        "E": "metabolic endocrine disorder management",
        "K": "gastrointestinal disease abdominal management",
        "F": "psychiatric mental health disorder",
        "G": "neurological disease brain diagnosis",
        "C": "malignancy cancer oncology management",
        "D": "hematological disorder blood count management",
        "S": "trauma injury orthopaedic fracture",
        "T": "poisoning overdose toxicological",
        "M": "musculoskeletal pain rheumatological",
        "Z": "preventive care follow-up clinical encounter",
        "Y": "external cause injury encounter",
        "B": "viral bacterial fungal infection management",
    }
    prefix = code[0].upper()
    return prefixes.get(prefix, f"ICD {code} clinical diagnosis management")


def download_pubmed_abstracts(
    codes: list,
    max_per_code: int = 30,
    email: str = "shifamind@research.com",
    sleep_between: float = 0.4,
) -> list:
    """
    Download PubMed abstracts for each ICD-10 code.

    Returns list of dicts: {"text": str, "diagnosis": str, "source": str,
                            "pmid": str, "title": str}
    """
    try:
        from Bio import Entrez
    except ImportError:
        log.error("biopython not installed. Run: pip install biopython")
        sys.exit(1)

    Entrez.email = email
    corpus = []
    total_downloaded = 0

    for code in codes:
        query = _CODE_TO_QUERY.get(code)
        if query is None:
            # Try prefix match
            for prefix in sorted(_CODE_TO_QUERY.keys(), key=len, reverse=True):
                if code.startswith(prefix):
                    query = _CODE_TO_QUERY[prefix]
                    break
        if query is None:
            query = _build_fallback_query(code)

        log.info(f"  {code:10s} → \"{query[:60]}\"")

        try:
            # Step 1: search
            handle  = Entrez.esearch(db="pubmed", term=query, retmax=max_per_code,
                                     sort="relevance")
            record  = Entrez.read(handle)
            handle.close()
            pmids   = record.get("IdList", [])
            time.sleep(sleep_between)

            if not pmids:
                log.warning(f"    No results for {code}")
                continue

            # Step 2: fetch abstracts
            handle  = Entrez.efetch(db="pubmed", id=",".join(pmids),
                                    rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            time.sleep(sleep_between)

            n_added = 0
            for article in records.get("PubmedArticle", []):
                try:
                    med   = article["MedlineCitation"]
                    pmid  = str(med["PMID"])
                    art   = med["Article"]
                    title = str(art.get("ArticleTitle", ""))
                    ab    = art.get("Abstract", {}).get("AbstractText", [])

                    # AbstractText can be a list of sections or a plain string
                    if isinstance(ab, list):
                        abstract = " ".join(str(s) for s in ab)
                    else:
                        abstract = str(ab)

                    if not abstract.strip():
                        continue

                    # Truncate to 600 chars — keep it focused and avoid padding
                    text = f"{title}. {abstract}"[:600]
                    corpus.append({
                        "text"     : text,
                        "diagnosis": code,
                        "source"   : "pubmed",
                        "pmid"     : pmid,
                        "title"    : title[:100],
                    })
                    n_added += 1
                except (KeyError, IndexError):
                    continue

            total_downloaded += n_added
            log.info(f"    Downloaded {n_added} abstracts  (total so far: {total_downloaded})")

        except Exception as e:
            log.warning(f"    NCBI fetch failed for {code}: {e}  — skipping")
            time.sleep(2)
            continue

    return corpus


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download PubMed abstracts for Top-50 ICD-10 codes")
    parser.add_argument("--max-per-code", type=int, default=30,
                        help="Max abstracts per ICD-10 code (default 30)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-download even if cache exists")
    parser.add_argument("--email", type=str, default=None,
                        help="Email for NCBI Entrez (overrides .env)")
    args = parser.parse_args()

    out_path = config.EVIDENCE_P3 / "pubmed_abstracts.json"
    config.EVIDENCE_P3.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.rebuild:
        log.info(f"PubMed cache already exists: {out_path.name}")
        with open(out_path) as f:
            existing = json.load(f)
        log.info(f"  {len(existing)} abstracts cached.  Use --rebuild to re-download.")
        return

    # Load top-50 codes
    if not config.TOP50_INFO_OUT.exists():
        log.error(f"top50_icd10_info.json not found: {config.TOP50_INFO_OUT}")
        log.error("Run phase1_train.py first.")
        sys.exit(1)

    with open(config.TOP50_INFO_OUT) as f:
        top50_info = json.load(f)
    codes = top50_info["top_50_codes"]
    log.info(f"Downloading PubMed abstracts for {len(codes)} ICD-10 codes …")
    log.info(f"Max per code: {args.max_per_code}")

    # Resolve email (NCBI strongly recommends providing one)
    email = args.email or getattr(config, "NCBI_EMAIL", None) or "shifamind@research.com"
    log.info(f"NCBI email: {email}")

    corpus = download_pubmed_abstracts(
        codes        = codes,
        max_per_code = args.max_per_code,
        email        = email,
        sleep_between= 0.4,   # 3 req/s limit unauthenticated
    )

    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)

    n_by_code = {}
    for doc in corpus:
        n_by_code[doc["diagnosis"]] = n_by_code.get(doc["diagnosis"], 0) + 1

    log.info(f"\nPubMed abstracts downloaded: {len(corpus)} total")
    log.info(f"Saved → {out_path}")
    log.info("Per-code breakdown:")
    for code, n in sorted(n_by_code.items(), key=lambda x: x[1]):
        log.info(f"  {code:10s}: {n:3d} abstracts")

    log.info("\nDone. Now run:")
    log.info("  python scripts/phase3_train.py --base-phase 1 --rebuild-corpus")
    log.info("  python scripts/phase3_train.py --base-phase 2 --rebuild-corpus")


if __name__ == "__main__":
    main()
