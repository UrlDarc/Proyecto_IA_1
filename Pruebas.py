import pandas as pd

# Cargar el dataset
df = pd.read_csv('Tweets.csv')

# Ver las primeras filas del dataset
df.head()

# Ver las columnas del dataset
df.columns

import re


# Reemplazar NaN por cadenas vacías en la columna 'text'
df['text'] = df['text'].fillna('')

# Definir la función de limpieza del texto
def clean_text(text):
    if not isinstance(text, str):  # Verifica si el valor es una cadena
        return ''  # Devuelve una cadena vacía si no lo es
    # Aquí puedes agregar las operaciones de limpieza (como eliminar caracteres especiales, convertir a minúsculas, etc.)
    text = text.lower()  # Convertir el texto a minúsculas
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'@\w+', '', text)  # Eliminar menciones (@usuario)
    text = re.sub(r'#\w+', '', text)  # Eliminar hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
    return text

# Aplicar la función de limpieza a la columna 'text'
df['cleaned_text'] = df['text'].apply(clean_text)

# Ver las primeras filas después de limpiar
print(df.head())

import nltk
nltk.download('punkt', download_dir=r"C:\Users\admin\Documents\David\Proyecto_IA\nltk_data")

nltk.download('punkt', download_dir=r"C:\Users\admin\Documents\David\Proyecto_IA\nltk_data")
nltk.download('stopwords', download_dir=r"C:\Users\admin\Documents\David\Proyecto_IA\nltk_data")
nltk.download('wordnet', download_dir=r"C:\Users\admin\Documents\David\Proyecto_IA\nltk_data")

print(nltk.data.path)

import nltk
import os
ruta_personalizada = "C:/Users/admin/Documents/David/Proyecto_IA/nltk_data"
nltk.download('punkt_tab', download_dir=ruta_personalizada)

nltk.data.path.clear()
nltk.data.path.append(ruta_personalizada)

# 3. Verifica que nltk "vea" el archivo
try:
    from nltk.data import find
    print("NLTK encontró:", find('tokenizers/punkt/english.pickle'))
except LookupError:
    print("No se encontró punkt. Intentando descargarlo...")

    # 4. Descargar si no se encuentra
    nltk.download('punkt', download_dir=ruta_personalizada)


    print("Reintentando...")
    from nltk.tokenize import word_tokenize
    text = "I`d have responded, if I were going"
    tokens = word_tokenize(text)
    print("Tokens:", tokens)
else:

    from nltk.tokenize import word_tokenize
    text = "I`d have responded, if I were going"
    tokens = word_tokenize(text)
    print("Tokens:", tokens)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Inicializar stopwords y lematizador
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Función para procesar texto: tokenizar, quitar stopwords y lematizar
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# Aplicar al DataFrame
df['lemmatized_text'] = df['cleaned_text'].apply(preprocess_text)

# Mostrar los resultados
print(df[['text', 'cleaned_text', 'lemmatized_text']].head())
