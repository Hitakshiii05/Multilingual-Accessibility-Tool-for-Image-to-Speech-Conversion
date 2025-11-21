# Multilingual Image → Text → LLM Translation → Speech (English & Hindi)

This project is a Google Colab–based demo that takes an **image with text** (e.g., medicine labels, notice boards, sign boards), runs **OCR** to extract the text, sends it to a **Large Language Model (LLM)** for **translation and explanation**, and finally converts the translated text into **speech**.

Supported languages in this implementation:
- **OCR**: English, Hindi  
- **LLM translation**: English ↔ Hindi  
- **Text-to-Speech**: English, Hindi  

The whole pipeline runs end-to-end inside a single **Google Colab notebook** with an interactive **Gradio UI**.

---

## ✨ Features

- 📷 **Image Input** – Upload any image containing English/Hindi text.
- 🔎 **OCR (Tesseract)** – Extracts raw text from the image.
- 🧠 **LLM Translation & Explanation (TinyLlama / FLAN-T5)**  
  - Translates text between English and Hindi.  
  - Optionally generates a **simple explanation** of the content.
- 🔊 **Text-to-Speech (gTTS)** – Speaks out the translated text in the target language.
- 🌐 **Gradio Web UI** – User-friendly interface directly inside Colab.
- 📂 **Small Custom Dataset** – Example English–Hindi instruction sentences used to show how the LLM runs over a dataset (and saves outputs as CSV).

---

## 🧱 System Architecture

High-level pipeline:

1. **Image Upload (Gradio UI)**  
   User uploads an image with English/Hindi text.

2. **OCR (Tesseract)**  
   The image is converted to a PIL image and passed to Tesseract:
   - Tesseract language model for English (`eng`)
   - Tesseract language model for Hindi (`hin`)
   Output: raw text string.

3. **LLM Translation & Explanation**  
   The OCR text is sent to a small LLM (e.g., `TinyLlama-1.1B-Chat-v1.0` or `google/flan-t5-small`) with a prompt that specifies:
   - Source language (English/Hindi)
   - Target language (English/Hindi)
   - Whether to also produce a simple explanation.

4. **Text-to-Speech (gTTS)**  
   The final translated text is fed into gTTS:
   - Language codes: `en` for English, `hi` for Hindi  
   Output: an `.mp3` audio file played inside the Gradio UI.

5. **Dataset Mode (Text-only)**  
   A small **English–Hindi dataset** is stored in a Pandas DataFrame.  
   For each row, the project:
   - Sends the sentence to the LLM
   - Stores the translated output in a new column
   - Exports results to `sample_en_hi_llm_translations.csv`

---

## 🛠️ Technologies Used

- **Python** (Google Colab)
- **OCR**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- **LLM**:  
  - Primary example: [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)  
  - Optionally: `google/flan-t5-small`
- **Hugging Face Transformers** (model & tokenizer loading, pipelines)
- **Text-to-Speech**: [`gTTS`](https://pypi.org/project/gTTS/)
- **UI**: [`Gradio`](https://gradio.app/)
- **Data Handling**: `pandas`, `numpy`
- **Imaging**: `Pillow (PIL)`

---

## 🚀 Getting Started (Google Colab)

### 1. Open the Notebook

- Upload the project notebook (e.g., `multilingual_llm_ocr_tts.ipynb`) to Google Drive.
- Open it in **Google Colab**.

### 2. Enable GPU (Recommended)

> Runtime → Change runtime type → **Hardware accelerator: GPU** → Save

GPU is recommended especially if you are using TinyLlama.  
The project also works on CPU with smaller models (like FLAN-T5).

### 3. Install Dependencies

Run the first cell in the notebook (example):

```python
!apt-get update -qq
!apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin

!pip install -q pytesseract Pillow transformers accelerate sentencepiece bitsandbytes gTTS gradio pandas
