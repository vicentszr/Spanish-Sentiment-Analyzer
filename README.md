# Spanish Sentiment Analyzer 🧠

BERT-based sentiment analysis for Spanish text, fine-tuned on 50,000 IMDB reviews.

## What it does
- Analyzes sentiment of any Spanish text (positive/negative)
- 88% accuracy on test set
- Applied to real restaurant reviews from Extremadura via Google Maps API

## Applications
- **FastAPI** — REST endpoint for developers
- **Streamlit** — visual interface to analyze Extremadura restaurants

## Stack
- BETO (Spanish BERT) — dccuchile/bert-base-spanish-wwm-cased
- HuggingFace Transformers
- FastAPI
- Streamlit
- Google Maps Places API

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 88% |
| F1 (positive) | 0.88 |
| F1 (negative) | 0.87 |
| Training data | 50,000 reviews |

## Run locally
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
python -m streamlit run app_streamlit.py
```

## Note
The trained model (beto_sentiment/) is not included due to file size.
Train it yourself using the notebook in Google Colab or download it from HuggingFace.
