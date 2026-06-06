import requests
import os
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

def get_reviews_google(nombre_restaurante, api_key):
    # Paso 1: buscar el lugar
    search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        'input': nombre_restaurante,
        'inputtype': 'textquery',
        'fields': 'place_id,name',
        'key': api_key
    }
    response = requests.get(search_url, params=params)
    data = response.json()
    print(data)

    if not data['candidates']:
        return []
    
    place_id = data['candidates'][0]['place_id']
    print(f"Restaurante encontrado: {data['candidates'][0]['name']}")
    
    # Paso 2: obtener reseñas
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        'place_id': place_id,
        'fields': 'name,reviews',
        'language': 'es',
        'key': api_key
    }
    response = requests.get(details_url, params=params)
    data = response.json()
    
    reseñas = []
    for review in data['result'].get('reviews', []):
        reseñas.append({
            'autor': review['author_name'],
            'texto': review['text'],
            'rating': review['rating']
        })
    
    return reseñas


load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')
reseñas = get_reviews_google('Restaurante Atrio Caceres', API_KEY)
print(f"Reseñas encontradas: {len(reseñas)}")
for r in reseñas:
    print(f"⭐{r['rating']} - {r['texto'][:100]}")

tokenizer = BertTokenizer.from_pretrained('beto_sentiment')
model = BertForSequenceClassification.from_pretrained('beto_sentiment', num_labels=2)
model.eval()

def analizar_sentimiento(texto):
    inputs = tokenizer(texto, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    probs = F.softmax(logits, dim=-1)
    confianza = probs[0][predicted_class].item()
    return 'positivo' if predicted_class == 1 else 'negativo', round(confianza * 100, 2)

# Analizar reseñas del Atrio
print("\n--- ANÁLISIS DE SENTIMIENTO ---")
for r in reseñas:
    sentimiento, confianza = analizar_sentimiento(r['texto'])
    print(f"⭐{r['rating']} | {sentimiento} ({confianza}%) | {r['texto'][:80]}")
    
st.divider()
st.subheader('🔍 Analizar texto directamente')
texto_libre = st.text_area('Escribe una reseña o comentario')
if st.button('Analizar texto'):
    if texto_libre:
        sentimiento, confianza = analizar_sentimiento(texto_libre)
        color = '🟢' if sentimiento == 'positivo' else '🔴'
        st.write(f"{color} {sentimiento} ({confianza}%)")