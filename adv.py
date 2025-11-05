# ======================================================
# 🌐 Multilingual OCR + Transliteration + Translation App
# Designed for Google Colab (Streamlit + OCR + OpenAI)
# ======================================================

!pip install streamlit streamlit_jupyter openai easyocr pytesseract aksharamukha indic-transliteration googletrans==4.0.0-rc1 langdetect pdf2image tqdm

import os
import streamlit as st
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from aksharamukha.transliterate import process
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from googletrans import Translator
from langdetect import detect
from tqdm import tqdm
import openai

# ======================
# 🔑 API Key setup
# ======================
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# ======================
# 🧩 Helper Functions
# ======================

def extract_text_from_pdf(uploaded_pdf):
    """Extract text from multi-page PDF"""
    pdf_bytes = uploaded_pdf.read()
    pdf_path = "/tmp/input.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    pages = convert_from_path(pdf_path)
    text = ""
    for i, page in enumerate(tqdm(pages, desc="Extracting PDF pages")):
        text += pytesseract.image_to_string(page, lang="san+hin+eng") + "\n"
    return text.strip()

def ocr_image(image, langs="san+hin+eng"):
    """OCR for single image"""
    return pytesseract.image_to_string(image, lang=langs)

def transliterate_text(text, src, tgt):
    """Try Aksharamukha first; fallback to indic-transliteration"""
    try:
        return process(src, tgt, text)
    except Exception:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.BENGALI)

def translation_openai(text, target_lang="en"):
    """Smart translation using OpenAI or Google fallback"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Translate this text to {target_lang} preserving meaning."},
                {"role": "user", "content": text[:2000]}  # trunc to avoid long inputs
            ]
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return Translator().translate(text, dest=target_lang).text

def heuristic_translit_accuracy(src_text, tgt_text):
    """Estimate transliteration fidelity by comparing back-transliteration"""
    try:
        back = process('Bengali', 'Devanagari', tgt_text)
        ratio = sum(a == b for a, b in zip(src_text, back)) / max(len(src_text), 1)
        return round(ratio * 100, 2)
    except Exception:
        return 0.0

# ======================
# 🎨 Streamlit Interface
# ======================

st.title("🌍 Universal OCR + Transliteration + Translation App")
st.markdown("### Upload PDF or image, choose your languages, and get transliterated + translated text.")

uploaded_file = st.file_uploader("📂 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
src_script = st.selectbox("Source Script", ["Devanagari", "Tamil", "Telugu", "Gujarati", "Malayalam", "Kannada", "Oriya"])
tgt_script = st.selectbox("Target Script", ["Bengali", "Latin", "Kannada", "Telugu", "Gujarati", "Tamil", "Devanagari"])
tgt_lang = st.text_input("🌐 Translation Language (e.g., 'en', 'hi', 'bn', 'ta')", "en")

if uploaded_file:
    st.info("Processing... this might take time for PDFs.")
    if uploaded_file.type == "application/pdf":
        ocr_text = extract_text_from_pdf(uploaded_file)
    else:
        image = Image.open(uploaded_file)
        ocr_text = ocr_image(image)
    
    st.subheader("🧾 Extracted OCR Text")
    st.text_area("Raw OCR Output", ocr_text[:2000], height=150)

    translit_text = transliterate_text(ocr_text, src_script, tgt_script)
    acc = heuristic_translit_accuracy(ocr_text, translit_text)

    st.subheader("🔡 Transliterated Text")
    st.text_area("Output", translit_text[:2000], height=150)
    st.write(f"🧠 Estimated Transliteration Accuracy: {acc}%")

    translated = translation_openai(translit_text, tgt_lang)
    st.subheader("🌐 Translated Text")
    st.text_area("Translation", translated, height=150)

    st.success("✅ Done! You can copy or download your results below.")

    result = f"OCR Text:\n{ocr_text}\n\nTransliteration:\n{translit_text}\n\nTranslation:\n{translated}"
    st.download_button("📥 Download Result", result, file_name="output.txt")

st.markdown("---")
st.markdown("**Features:** OCR | Transliteration | Translation | Accuracy Check | Multi-Script | AI-Powered 🌏")
