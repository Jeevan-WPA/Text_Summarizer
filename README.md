# 📄 Text Summarizer

A a dual-mode summarization system for **Extractive** and **Abstractive** text summarization. Also implemented a **GPT-powered** text summarization — now with a **Streamlit UI**.

This project includes:

* Classical extractive summarization
* T5-based abstractive summarization
* GPT summarizer using OpenAI API
* FastAPI backend
* Streamlit web interface (ready to run)
* Training notebooks for fine-tuning
* ROUGE comparison endpoint

## 🚀 Features

### 🔹 Extractive Summarization

Selects the most important sentences using classical NLP techniques.
Fast, factual, and lightweight.

### 🔹 Abstractive Summarization

Powered by **T5 (fine-tuned)**, generating human-like summaries.
Full training pipeline included.

### 🔹 GPT Summarization

Uses OpenAI GPT models for high-quality, natural text.
Supports adjustable creativity & summary length.

### 🔹 Streamlit UI (NEW)

A simple, clean, user-friendly interface that allows anyone to:

* Paste text
* Choose summarization mode
* Get instant results
* Compare outputs side-by-side

Located in:
`text_summarizer_ui/`

## 🗂 Repository Structure

```
Text_Summarizer/
│
├── api.py                         # FastAPI backend service
├── extractive.py                  # Extractive summarizer
├── abstractive.py                 # T5 based summarizer
├── gpt_summarizer.py              # GPT (OpenAI) summarizer
│
├── t5-small-final                 # Consists of the Model files after training
│
├── text_summarizer_ui/            # Streamlit UI app
│   ├── app.py                     # Main Streamlit interface
│   └── api_client.py              # This file connects Streamlit → FastAPI.
│
├── t5_abstractive_train.ipynb     # T5 Fine-tuning notebook
│
├── requirements.txt               # Dependencies
└── README.md
```

## 🔧 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Jeevan-WPA/Text_Summarizer.git
cd Text_Summarizer
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create a .env file and add your API key

```bash
API_KEY=your_api_key_here
```


# 🌐 Running the API (Backend)

Start FastAPI server:

```bash
uvicorn api:app --reload
```

## Endpoints:

| Method | Endpoint       | Description                                          |
| ------ | -------------- | -----------------------------------------------------|
| POST   | `/extractive`  | Extractive summary                                   |
| POST   | `/abstractive` | Neural abstractive summary                           |
| POST   | `/compare`     | Returns extractive and abstractive summaries + ROUGE |
| POST   | `/gpt`         | GPT summary                                          |

# 🖥️ Running the Streamlit UI 

The UI is inside `text_summarizer_ui/`.

### Run the app:

```bash
cd text_summarizer_ui
streamlit run app.py
```






