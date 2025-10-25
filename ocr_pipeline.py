#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import List


# ---- Configure paths ----
BASE_DIR = Path("/Users/sahithikaruparthi/Desktop/ocrToDB")
PDF_DIR = BASE_DIR / "Biology"
WORK_DIR = BASE_DIR / ".dolphin_work"
MD_OUT_DIR = BASE_DIR / "markdown"

# Dolphin repo and HF model
DOLPHIN_REPO_DIR = Path("/Users/sahithikaruparthi/Desktop/gradeBoostRAG/pre-processing-pipeline/Dolphin")
HF_MODEL_DIR = DOLPHIN_REPO_DIR / "hf_model"

if not DOLPHIN_REPO_DIR.exists():
    raise SystemExit(f"Dolphin repo not found at {DOLPHIN_REPO_DIR}")
if not HF_MODEL_DIR.exists():
    raise SystemExit(f"Dolphin hf_model not found at {HF_MODEL_DIR}")

# Ensure we can import Dolphin modules
if str(DOLPHIN_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(DOLPHIN_REPO_DIR))


def ensure_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dolphin_model():
    # Use the Hugging Face variant for simpler local loading
    from demo_page_hf import DOLPHIN  

    model = DOLPHIN(str(HF_MODEL_DIR))
    return model


def dolphin_parse_pdf(model, pdf_path: Path):
    """Use Dolphin's HF document pipeline to parse a PDF into recognition results."""
    from demo_page_hf import process_document 

    save_dir = str(WORK_DIR)
    json_path, recognition_results = process_document(
        document_path=str(pdf_path),
        model=model,
        save_dir=save_dir,
        max_batch_size=16,
    )
    return json_path, recognition_results


def dolphin_results_to_markdown(recognition_results: list) -> str:
    from utils.markdown_utils import MarkdownConverter  # type: ignore

    # recognition_results from HF path is a list of pages with "elements"
    # Flatten if needed
    if recognition_results and isinstance(recognition_results[0], dict) and "elements" in recognition_results[0]:
        flat = []
        for page in recognition_results:
            flat.extend(page.get("elements", []))
        recognition_results = flat

    converter = MarkdownConverter()
    md = converter.convert(recognition_results)
    return md


def write_markdown(md_text: str, pdf_stem: str) -> Path:
    md_path = MD_OUT_DIR / f"{pdf_stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


def main():
    ensure_dirs()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        sys.exit(1)

    print("Loading Dolphin (HF) model ...")
    model = load_dolphin_model()

    produced_mds: List[Path] = []
    for pdf in pdf_files:
        print(f"Processing {pdf.name} ...")
        _, recognition_results = dolphin_parse_pdf(model, pdf)
        md_text = dolphin_results_to_markdown(recognition_results)
        md_path = write_markdown(md_text, pdf.stem)
        print(f"  -> {md_path}")
        produced_mds.append(md_path)



if __name__ == "__main__":
    main()
