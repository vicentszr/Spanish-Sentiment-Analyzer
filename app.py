from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained('beto_sentiment')
model = BertForSequenceClassification.from_pretrained('beto_sentiment', num_labels=2)
model.eval()

class Review(BaseModel):
    texto: str

@app.post('/predecir')
def predecir(review: Review):
    # aquí va la predicción
    inputs = tokenizer(review.texto, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()


    probs = F.softmax(logits, dim=-1)
    confianza = probs[0][predicted_class].item()
    etiqueta = 'positivo' if predicted_class == 1 else 'negativo'

    return {
        'prediccion': etiqueta,
        'confianza': round(confianza * 100, 2)
    }