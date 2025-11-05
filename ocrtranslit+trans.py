# app_translit_ocr_translate.py
import os
import io
import re
import zipfile
import tempfile
import subprocess
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from PIL import Image
from pdf2image import convert_from_path

# Transliteration
from aksharamukha.transliterate import process as ak_process

# Optional libs
try:
    import pytesseract
    TESSERACT_PY_INSTALLED = True
except Exception:
    TESSERACT_PY_INSTALLED = False

try:
    import easyocr
    EASYOCR_INSTALLED = True
except Exception:
    EASYOCR_INSTALLED = False

try:
    import openai
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False

# indic-transliteration (fallback for some roman schemes)
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
    INDIC_INSTALLED = True
except Exception:
    INDIC_INSTALLED = False

# ---------------------------
# Helper detection utilities
# ---------------------------
def is_tesseract_available() -> bool:
    """Return True if tesseract binary is installed and runnable (subprocess ok)."""
    if not TESSERACT_PY_INSTALLED:
        return False
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

TESSERACT_AVAILABLE = is_tesseract_available()

# ---------------------------
# OCR functions
# ---------------------------
def ocr_with_tesseract_pil(img: Image.Image, lang: str = "eng") -> str:
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as e:
        return f"[TESSERACT OCR ERROR] {e}"

def ocr_with_easyocr_path(img_path: str, reader: "easyocr.Reader") -> str:
    # reader.readtext returns list of strings if detail=0
    try:
        results = reader.readtext(img_path, detail=0)
        return " ".join(results)
    except Exception as e:
        return f"[EASYOCR ERROR] {e}"

# ---------------------------
# File processing helpers
# ---------------------------
def pdf_to_images(pdf_bytes: bytes, dpi: int = 200, output_folder: Optional[str] = None) -> List[str]:
    """Convert PDF bytes to images and return paths to images."""
    if output_folder is None:
        tmpdir = tempfile.mkdtemp()
    else:
        tmpdir = output_folder
    pdf_path = os.path.join(tmpdir, "tmp_upload.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    # convert_from_path writes PIL Image objects; we will save them as jpg files and return paths
    pages = convert_from_path(pdf_path, dpi=dpi, output_folder=tmpdir)
    paths = []
    for i, page in enumerate(pages, start=1):
        p = os.path.join(tmpdir, f"page_{i:04d}.jpg")
        page.save(p, "JPEG")
        paths.append(p)
    return paths

def extract_zip_to_temp(zip_bytes: bytes) -> List[str]:
    tmpdir = tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as zf:
        zf.write(zip_bytes)
        zf.flush()
        with zipfile.ZipFile(zf.name, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
    # gather images and pdfs (sorted)
    files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir)])
    # flatten pdfs into images
    paths = []
    for f in files:
        lower = f.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            paths.append(f)
        elif lower.endswith(".pdf"):
            try:
                img_paths = pdf_to_images(open(f, "rb").read(), output_folder=tmpdir)
                paths.extend(img_paths)
            except Exception as e:
                # skip problematic pdf
                paths.append(f"[PDF_CONVERT_ERROR: {e}]")
    # filter valid image paths
    paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
    return paths

# ---------------------------
# High-level OCR pipeline
# ---------------------------
def ocr_paths_sequential(image_paths: List[str], use_tesseract: bool, ocr_lang: str, easy_reader=None, max_workers:int=4):
    """OCR list of image paths sequentially (or multithreaded) and return text per path."""
    results = {}
    total = len(image_paths)
    if total == 0:
        return results
    # Use ThreadPool to speed up where IO/processing allows
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for p in image_paths:
            if use_tesseract:
                # load image and call tesseract
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    results[p] = f"[OPEN ERROR] {e}"
                    continue
                futures[exe.submit(ocr_with_tesseract_pil, img, ocr_lang)] = p
            else:
                # easyocr path uses file path and reader
                futures[exe.submit(ocr_with_easyocr_path, p, easy_reader)] = p

        done = 0
        for fut in concurrent.futures.as_completed(futures):
            p = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                text = f"[OCR THREAD ERROR] {e}"
            results[p] = text
            done += 1
            st.session_state["progress"].progress(done / total)
    return results

# ---------------------------
# Transliteration (Aksharamukha + indic fallback)
# ---------------------------
def transliterate_aksharamukha(src_script: str, tgt_script: str, text: str) -> str:
    try:
        return ak_process(src_script, tgt_script, text)
    except Exception as e:
        return f"[AKSHARAMUKHA ERROR] {e}"

def transliterate_indic(src_scheme: str, tgt_scheme: str, text: str) -> str:
    if not INDIC_INSTALLED:
        return "[INDIC-TRANSLITERATION NOT INSTALLED]"
    try:
        return indic_transliterate(text, getattr(sanscript, src_scheme), getattr(sanscript, tgt_scheme))
    except Exception as e:
        return f"[INDIC TRANSLIT ERROR] {e}"

# ---------------------------
# OpenAI Translation/Refinement
# ---------------------------
def openai_translate(text: str, target_language: str, system_prompt: str = "") -> Optional[str]:
    if not OPENAI_INSTALLED or not getattr(openai, "api_key", None):
        return None
    prompt = f"{system_prompt}\nTranslate the following text into {target_language}. Keep named entities as-is.\n\nText:\n{text}"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt or "You are a helpful translator."},
                      {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=4000
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OPENAI ERROR] {e}"

# ---------------------------
# Utilities
# ---------------------------
def clean_text_for_translit(text: str) -> str:
    # remove excessive whitespace and weird control chars
    return re.sub(r'\s+', ' ', text).strip()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ---------------------------
# Streamlit UI & flow
# ---------------------------
st.set_page_config(page_title="Universal OCR → Transliterate → Translate", layout="wide")
st.title("Universal OCR → Transliteration → Translation Studio")
st.markdown("Upload files (txt / image / pdf / zip of images/pdfs). Choose OCR language, transliteration target script, and translation language (OpenAI).")

# Sidebar controls
st.sidebar.header("Settings")
ocr_lang = st.sidebar.text_input("OCR language code(s) (Tesseract/EasyOCR)", value="eng")
use_tesseract_checkbox = st.sidebar.checkbox("Prefer Tesseract (if available)", value=True)
use_tesseract = use_tesseract_checkbox and TESSERACT_AVAILABLE
if use_tesseract_checkbox and not TESSERACT_AVAILABLE:
    st.sidebar.warning("Tesseract not available in this environment; EasyOCR fallback will be used.")

translit_mode = st.sidebar.selectbox("Transliteration Engine", options=["Aksharamukha","Indic-Transliteration"], index=0)
# For aksharamukha supported scripts list - a useful subset (user can type exact names)
akshara_hint = "Examples for Aksharamukha: Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati, Arabic, Hebrew"
src_script_input = st.sidebar.text_input("Source script (Aksharamukha name / indic scheme)", value="Devanagari", help=akshara_hint)
tgt_script_input = st.sidebar.text_input("Target script (Aksharamukha name / indic scheme)", value="Bengali", help=akshara_hint)

enable_openai = st.sidebar.checkbox("Enable OpenAI translation/refine (requires OPENAI_API_KEY)", value=False)
openai_key = st.sidebar.text_input("OpenAI API key (optional; will override env var)", type="password")
if enable_openai and openai_key:
    openai.api_key = openai_key
elif enable_openai and getattr(openai, "api_key", None) is None:
    # try environment
    if os.getenv("OPENAI_API_KEY") is None:
        st.sidebar.warning("Set OPENAI_API_KEY env var or paste key in the box above to use OpenAI.")

translation_target_lang = st.sidebar.text_input("Translation target language (e.g., English, Bengali)", value="English")
concurrency = st.sidebar.slider("OCR concurrency (threads)", 1, 8, 4)
do_translation = st.sidebar.checkbox("Also translate (OpenAI)", value=False)

# File uploader
st.header("Upload files")
uploaded = st.file_uploader("Upload one or more (txt / pdf / png / jpg / zip)", accept_multiple_files=True)

# initialize progress container
if "progress" not in st.session_state:
    st.session_state["progress"] = st.empty()
    st.session_state["progress_bar"] = st.progress(0.0)

# Processing
if uploaded:
    start_time = time.time()
    outputs = {}  # filename -> dict with fields: ocr_text, translit_text, translation
    # Prepare easyocr reader only if needed
    easy_reader = None
    if not use_tesseract and EASYOCR_INSTALLED:
        # EasyOCR supports combined language list; for Bengali we should include 'en' in list
        langs = [lang.strip() for lang in ocr_lang.split("+") if lang.strip()]
        # if 'bn' present, easyocr recommends ['bn','as','en']
        if "bn" in langs and "en" not in langs:
            langs = ["bn","as","en"]
        easy_reader = easyocr.Reader(lang_list=langs, gpu=False)

    # loop files
    total_files = len(uploaded)
    file_index = 0
    for f in uploaded:
        file_index += 1
        fname = f.name
        st.write(f"Processing file {file_index}/{total_files}: {fname}")
        b = f.read()
        extracted_paths = []
        ocr_results = {}

        # determine file type
        lower = fname.lower()
        if lower.endswith(".txt"):
            try:
                text = b.decode("utf-8")
            except:
                text = b.decode("latin-1", errors="ignore")
            clean = clean_text_for_translit(text)
            ocr_results["single_text"] = clean
            extracted_paths = []  # no images
        elif lower.endswith(".pdf"):
            # convert pdf to images
            try:
                st.info("Converting PDF to images (may take a while)...")
                image_paths = pdf_to_images(b, dpi=200)
                extracted_paths = image_paths
            except Exception as e:
                st.error(f"PDF -> images failed: {e}")
                outputs[fname] = {"error": f"PDF conversion error: {e}"}
                continue
        elif lower.endswith(".zip"):
            try:
                st.info("Extracting ZIP and collecting pages...")
                image_paths = extract_zip_to_temp(b)
                if not image_paths:
                    st.warning("No image pages were found in ZIP.")
                extracted_paths = image_paths
            except Exception as e:
                st.error(f"ZIP extract error: {e}")
                outputs[fname] = {"error": f"ZIP extract error: {e}"}
                continue
        elif lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            # save temporary image file
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1])
            tmpf.write(b); tmpf.flush(); tmpf.close()
            extracted_paths = [tmpf.name]
        else:
            st.warning(f"Unsupported file type: {fname} — skipping.")
            continue

        # If we have image pages to OCR
        if extracted_paths:
            # init progress
            st.session_state["progress"].progress(0.0)
            st.info(f"Running OCR on {len(extracted_paths)} pages...")
            ocr_map = ocr_paths_sequential(extracted_paths, use_tesseract, ocr_lang, easy_reader, max_workers=concurrency)
            # Merge texts in filename order
            ordered = sorted(ocr_map.keys())
            merged = "\n\n".join([f"\n--- {os.path.basename(p)} ---\n{ocr_map[p]}" for p in ordered])
            clean = clean_text_for_translit(merged)
            ocr_results["single_text"] = clean
        # Now transliterate
        source_text = ocr_results.get("single_text", "")
        translit_text = ""
        if source_text:
            if translit_mode == "Aksharamukha":
                try:
                    translit_text = transliterate_aksharamukha(src_script_input, tgt_script_input, source_text)
                except Exception as e:
                    translit_text = f"[AKSHARAMUKHA ERROR] {e}"
            else:
                # try indic-transliteration fallback (assuming src/tgt are sanscript names)
                translit_text = transliterate_indic(src_script_input, tgt_script_input, source_text)
        else:
            translit_text = "[NO TEXT FOUND]"

        # Optionally translate via OpenAI
        translation_text = None
        if do_translation and enable_openai and OPENAI_INSTALLED and getattr(openai, "api_key", None):
            st.info("Calling OpenAI for translation (may cost tokens)...")
            translation_text = openai_translate(translit_text if translit_mode=="Aksharamukha" else source_text, translation_target_lang,
                                                system_prompt="You are a careful translator. Keep named entities unchanged.")
        # Save outputs
        outputs[fname] = {
            "ocr_text": source_text,
            "transliteration": translit_text,
            "translation": translation_text
        }

        # show previews + download
        st.subheader(f"Result preview — {fname}")
        st.markdown("**OCR text (sample)**")
        st.text_area(f"ocr_{fname}", source_text[:4000], height=180)
        st.markdown(f"**Transliteration → {tgt_script_input} (sample)**")
        st.text_area(f"translit_{fname}", translit_text[:4000], height=180)
        if translation_text:
            st.markdown(f"**Translation → {translation_target_lang} (sample)**")
            st.text_area(f"translate_{fname}", translation_text[:4000], height=180)

        # per-file downloads
        st.download_button(f"Download OCR ({fname})", data=source_text.encode("utf-8"), file_name=f"{fname}__ocr.txt", mime="text/plain")
        st.download_button(f"Download Transliteration ({fname})", data=translit_text.encode("utf-8"), file_name=f"{fname}__translit.txt", mime="text/plain")
        if translation_text:
            st.download_button(f"Download Translation ({fname})", data=translation_text.encode("utf-8"), file_name=f"{fname}__translation.txt", mime="text/plain")

        # reset progress for next file
        st.session_state["progress"].progress(0.0)

    # End loop
    # allow zip download of all outputs
    zipbuf = io.BytesIO()
    with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, parts in outputs.items():
            base = os.path.splitext(fname)[0]
            zf.writestr(f"{base}__ocr.txt", parts.get("ocr_text","").encode("utf-8"))
            zf.writestr(f"{base}__translit.txt", parts.get("transliteration","").encode("utf-8"))
            if parts.get("translation") is not None:
                zf.writestr(f"{base}__translation.txt", parts.get("translation","").encode("utf-8"))
    zipbuf.seek(0)
    st.download_button("Download ALL outputs (ZIP)", data=zipbuf, file_name="translit_outputs.zip", mime="application/zip")

    elapsed = time.time() - start_time
    st.success(f"Done. Processed {len(outputs)} files in {elapsed:.1f}s")

else:
    st.info("Upload files to start. Use sidebar to configure OCR engine, scripts, and OpenAI options.")

# Footer note
st.markdown("---")
st.markdown("Tips: For best OCR on local machine install tesseract and poppler. In cloud/Colab, EasyOCR fallback will run. Aksharamukha supports many scripts. OpenAI translation is optional and requires a valid API key.")
