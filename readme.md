# Python packages
pip install streamlit pillow pdf2image aksharamukha easyocr indic-transliteration openai langdetect

# System (Ubuntu/Debian) for best performance
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
# Optional language packs for Tesseract, e.g.:
sudo apt-get install -y tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-eng

⚙️ How to Run in Google Colab

Open Google Colab

Paste the above script in a cell

Run this cell once

Then run:

!streamlit run /content/app.py --server.port 8501

advanced features

| Feature                      | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| Multi-language OCR           | Supports Devanagari, Tamil, Bengali, Kannada, etc.  |
| Multi-script Transliteration | Aksharamukha + Indic-transliteration fallback       |
| Smart Translation            | Uses OpenAI (GPT-4-mini) or Google Translate        |
| Accuracy Check               | Heuristic back-conversion scoring                   |
| PDF Batch Support            | Sequential OCR of 100+ pages                        |
| GPU Support                  | Uses EasyOCR if GPU available                       |
| Streamlit UI                 | Clean interface for uploads, options, and downloads |
| Caching                      | Reduces repeated processing time                    |
| Logging & Progress Bars      | Shows OCR progress in Colab                         |
| Portable                     | Works with any modern Colab instance                |

