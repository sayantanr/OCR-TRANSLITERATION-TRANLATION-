# ==========================================================
# 🌐 UNIVERSAL OCR + TRANSLITERATION + TRANSLATION (COLAB)
# Supports: images, PDFs, ZIPs of images
# OCR via EasyOCR (works in Colab)
# Transliteration via Aksharamukha
# Translation via OpenAI API (optional)
# ==========================================================

!pip install easyocr aksharamukha pdf2image openai Pillow tqdm langdetect --quiet
!apt-get -y install poppler-utils > /dev/null

import os, io, zipfile, tempfile, re
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path
import easyocr
from aksharamukha.transliterate import process as aksharamukha_process

# Optional
import openai
from langdetect import detect

# =============== CONFIG SECTION ===============
# 👇 Enter your OpenAI key if you want translation
openai.api_key = ""   # ← Paste here (or leave blank to skip translation)
TRANSLATE = False if openai.api_key == "" else True

# OCR and Transliteration parameters
OCR_LANGS = ['hi','en']       # e.g. ['hi','en'] for Devanagari text
SRC_SCRIPT = "Devanagari"     # e.g. 'Devanagari', 'Tamil', 'Roman'
TGT_SCRIPT = "Bengali"        # e.g. 'Bengali', 'Telugu', etc.
TARGET_TRANSLATION_LANG = "English"
# ==============================================

# Initialize EasyOCR reader (cached model download)
reader = easyocr.Reader(OCR_LANGS, gpu=False)

# Helper: PDF → image paths
def pdf_to_images(pdf_bytes):
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "temp.pdf")
    with open(pdf_path, "wb") as f: f.write(pdf_bytes)
    pages = convert_from_path(pdf_path, dpi=200)
    img_paths = []
    for i, page in enumerate(pages):
        pth = os.path.join(tmpdir, f"page_{i+1}.jpg")
        page.save(pth, "JPEG")
        img_paths.append(pth)
    return img_paths

# Helper: extract zip of pdf/images
def extract_zip(zbytes):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zbytes), 'r') as zf:
        zf.extractall(tmpdir)
    all_files = []
    for root,_,files in os.walk(tmpdir):
        for f in files:
            full = os.path.join(root,f)
            if f.lower().endswith((".jpg",".jpeg",".png",".pdf")):
                all_files.append(full)
    return all_files

# OCR routine
def run_ocr(path):
    if path.lower().endswith(".pdf"):
        pages = pdf_to_images(open(path,'rb').read())
        texts = []
        for p in pages:
            texts.append(" ".join(reader.readtext(p, detail=0)))
        return "\n".join(texts)
    else:
        return " ".join(reader.readtext(path, detail=0))

# Transliteration
def transliterate_text(text, src=SRC_SCRIPT, tgt=TGT_SCRIPT):
    try:
        return aksharamukha_process(src, tgt, text)
    except Exception as e:
        return f"[Transliteration error: {e}]"

# Optional translation
def translate_text(text, target_lang=TARGET_TRANSLATION_LANG):
    if not TRANSLATE or not openai.api_key:
        return "[Translation skipped]"
    try:
        msg = [{"role":"system","content":"You are a helpful translator."},
               {"role":"user","content":f"Translate the following text into {target_lang}:\n{text}"}]
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=msg, temperature=0)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Translation error: {e}]"

# =============== Upload Section ===============
from google.colab import files
print("📤 Upload PDF(s), image(s), or ZIPs (of images/pdfs):")
uploads = files.upload()
filelist = list(uploads.keys())
print(f"\n✅ Uploaded {len(filelist)} files\n")

# =============== Processing ===============
all_outputs = []
for fname, b in tqdm(uploads.items()):
    tempf = os.path.join(tempfile.gettempdir(), fname)
    open(tempf, "wb").write(b)
    filepaths = []

    if fname.lower().endswith(".zip"):
        filepaths = extract_zip(b)
    else:
        filepaths = [tempf]

    ocr_text_total = ""
    for f in filepaths:
        ocr_text_total += "\n\n" + run_ocr(f)

    # Transliterate
    translit_text = transliterate_text(ocr_text_total, SRC_SCRIPT, TGT_SCRIPT)

    # Translate
    translated = translate_text(translit_text, TARGET_TRANSLATION_LANG)

    # Save results
    out_base = os.path.splitext(fname)[0]
    with open(out_base + "_OCR.txt", "w", encoding="utf-8") as f: f.write(ocr_text_total)
    with open(out_base + "_Translit.txt", "w", encoding="utf-8") as f: f.write(translit_text)
    with open(out_base + "_Translated.txt", "w", encoding="utf-8") as f: f.write(translated)
    all_outputs.append((out_base + "_OCR.txt", out_base + "_Translit.txt", out_base + "_Translated.txt"))

print("\n✅ All processing done!\n")

# =============== Download Section ===============
out_zip = "colab_ocr_translit_results.zip"
with zipfile.ZipFile(out_zip, "w") as zf:
    for trio in all_outputs:
        for f in trio:
            zf.write(f)
files.download(out_zip)
print(f"📦 Results ready: {out_zip}")
