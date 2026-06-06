import streamlit as st
from scraper import get_reviews_google, analizar_sentimiento

st.title('🍽️ Analizador de Restaurantes Extremeños')
st.write('Analiza el sentimiento de las reseñas de Google Maps con IA')

API_KEY = 'AIzaSyBCy4Jx0ZtIEZbarK_YusAiRwqhbK9GpHw'

restaurante = st.text_input('Nombre del restaurante', 'Restaurante Atrio Caceres')

if st.button('Analizar'):
    with st.spinner('Buscando reseñas...'):
        reseñas = get_reviews_google(restaurante, API_KEY)
    
    if not reseñas:
        st.error('No se encontraron reseñas')
    else:
        st.success(f'Se encontraron {len(reseñas)} reseñas')
        
        positivas = 0
        negativas = 0
        resultados = []
        
        for r in reseñas:
            sentimiento, confianza = analizar_sentimiento(r['texto'])
            if sentimiento == 'positivo':
                positivas += 1
            else:
                negativas += 1
            resultados.append((r, sentimiento, confianza))
        
        col1, col2, col3 = st.columns(3)
        col1.metric('Total reseñas', len(reseñas))
        col2.metric('🟢 Positivas', f'{positivas} ({round(positivas/len(reseñas)*100)}%)')
        col3.metric('🔴 Negativas', f'{negativas} ({round(negativas/len(reseñas)*100)}%)')
        
        recomendacion = '✅ Recomendado' if positivas/len(reseñas) >= 0.7 else '⚠️ Con reservas' if positivas/len(reseñas) >= 0.5 else '❌ No recomendado'
        st.subheader(recomendacion)
        st.divider()
        
        for r, sentimiento, confianza in resultados:
            color = '🟢' if sentimiento == 'positivo' else '🔴'
            st.write(f"{color} ⭐{r['rating']} | {sentimiento} ({confianza}%)")
            st.caption(r['texto'])
            st.divider()
st.divider()
st.subheader('🔍 Analizar texto directamente')
texto_libre = st.text_area('Escribe una reseña o comentario')
if st.button('Analizar texto'):
    if texto_libre:
        sentimiento, confianza = analizar_sentimiento(texto_libre)
        color = '🟢' if sentimiento == 'positivo' else '🔴'
        st.write(f"{color} {sentimiento} ({confianza}%)")