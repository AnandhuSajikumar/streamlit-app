import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import openai
import time

import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Load and process data
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Standardize nutrition: Assume data has 'nutrition' as dict with keys like 'calories', 'protein'
        df['nutrition_str'] = df['nutrition'].apply(lambda x: f"Calories: {x.get('calories', 0)}kcal, Protein: {x.get('protein', 0)}g, Carbs: {x.get('carbs', 0)}g, Fat: {x.get('fat', 0)}g")
        df['full_text'] = df.apply(lambda row: f"Recipe: {row.get('name', 'Unknown')}\nIngredients: {', '.join(row.get('ingredients', []))}\nInstructions: {row.get('instructions', 'N/A')}\nNutrition: {row['nutrition_str']}\nCuisine: {row.get('cuisine', 'General')}", axis=1)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Ingest into Chroma
def ingest_to_chroma(df):
    try:
        client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        collection = client.get_or_create_collection(name="recipes")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        for idx, row in df.iterrows():
            embedding = model.encode(row['full_text']).tolist()
            collection.add(
                documents=[row['full_text']],
                metadatas=[{"name": row.get('name', 'Unknown'), "cuisine": row.get('cuisine', 'General')}],
                ids=[str(idx)],
                embeddings=[embedding]
            )
        client.persist()
        return collection, model
    except Exception as e:
        print(f"Error in ingest_to_chroma: {e}")
        raise

# Retrieve and Generate
def rag_query(query, collection, model, dietary_restrictions, health_condition, nutritional_goals):
    try:
        start_time = time.time()
        augmented_query = f"{query} {dietary_restrictions} for {health_condition}. Optimize for {nutritional_goals}."
        query_embedding = model.encode(augmented_query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No relevant recipes found."
        
        latency = time.time() - start_time
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Suggest recipes, substitutions, and track goals based on context."},
                {"role": "user", "content": f"Context: {context}\nQuery: {augmented_query}\nProvide: 1. Recommended recipes. 2. Substitutions for restrictions/allergies. 3. Nutritional analysis. 4. Why it matches health condition."}
            ]
        )
        generated = response.choices[0].message.content
        
        accuracy = 1.0 if results['documents'] and any(term in context.lower() for term in dietary_restrictions.lower().split()) else 0.5
        
        return generated, latency, accuracy, results.get('metadatas', [{}])[0]
    except Exception as e:
        print(f"Error in rag_query: {e}")
        raise