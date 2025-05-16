# TDS Virtual TA

A Virtual Teaching Assistant built for the Tools in Data Science course (IITM BSc Data Science), capable of answering student questions using scraped Discourse forum posts and course content.

---

## 🚀 Features

* FastAPI-powered REST API endpoint
* Accepts POST requests with student questions and optional image (base64) input
* Converts attached screenshots to text using OCR (pytesseract)
* Uses FAISS + OpenAI embeddings to search contextually relevant chunks
* Generates rubric-aware answers using GPT-3.5-Turbo via AI Proxy
* Returns both an `answer` and source `links`

---

## 📦 Requirements

* Python 3.9+
* Dependencies in `requirements.txt`
* AIPIPE\_TOKEN (for AIProxy OpenAI-compatible endpoint)

---

## 📁 Project Structure

```bash
.
├── app.py                        # FastAPI app
├── chunked_with_embeddings.json # Embedding output
├── chunk_metadata.json          # Metadata used by FAISS
├── faiss.index                  # FAISS vector index
├── scraper.py                   # Discourse scraping script
├── .env.example                 # Sample env file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🔧 .env Setup

Create a `.env` file based on the template below:

```env
AIPIPE_TOKEN=your_actual_token_here
```

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 🔍 Sample curl request:

```bash
curl -X POST http://localhost:8000/api/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I deploy FastAPI?", "image": null}'
```

---

## 🌐 Deployment

This app is deployed on [Render](https://render.com) and publicly accessible at:

```
https://tds-virtual-ta.onrender.com/api/
```

---

## 🧠 Promptfoo Evaluation Ready

The app is compatible with [Promptfoo](https://promptfoo.dev/) rubric testing. See `promptfooconfig.yaml` to evaluate with realistic test cases.

---


## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🤝 Contributions

Pull requests welcome! For major changes, open an issue first to discuss what you’d like to change.

---

## ✍️ Author

* **Abhijith VS**
  [GitHub](https://github.com/VS-Abhijith) | [LinkedIn](https://www.linkedin.com/in/vsabhijith)

---

