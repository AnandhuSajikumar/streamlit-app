import pandas as pd
import json
import time
import openai
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configure OpenRouter API
# -----------------------------
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"  # Important change

# -----------------------------
# Load and process recipe data
# -----------------------------
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Standardize nutrition
        df['nutrition_str'] = df['nutrition'].apply(
            lambda x: f"Calories: {x.get('calories', 0)}kcal, "
                      f"Protein: {x.get('protein', 0)}g, "
                      f"Carbs: {x.get('carbs', 0)}g, "
                      f"Fat: {x.get('fat', 0)}g"
        )

        # Combine into full text
        df['full_text'] = df.apply(
            lambda row: f"Recipe: {row.get('name', 'Unknown')}\n"
                        f"Ingredients: {', '.join(row.get('ingredients', []))}\n"
                        f"Instructions: {row.get('instructions', 'N/A')}\n"
                        f"Nutrition: {row['nutrition_str']}\n"
                        f"Cuisine: {row.get('cuisine', 'General')}",
            axis=1
        )

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# -----------------------------
# Ingest into ChromaDB
# -----------------------------
def ingest_to_chroma(df):
    try:
        client = PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="recipes")

        model = SentenceTransformer('all-MiniLM-L6-v2')

        for idx, row in df.iterrows():
            embedding = model.encode(row['full_text']).tolist()
            collection.add(
                documents=[row['full_text']],
                metadatas=[{
                    "name": row.get('name', 'Unknown'),
                    "cuisine": row.get('cuisine', 'General')
                }],
                ids=[str(idx)],
                embeddings=[embedding]
            )

        return collection, model
    except Exception as e:
        print(f"Error in ingest_to_chroma: {e}")
        raise

# -----------------------------
# Retrieve & Generate
# -----------------------------
def rag_query(query, collection, model, dietary_restrictions, health_condition, nutritional_goals):
    try:
        start_time = time.time()

        # Augment query with context
        augmented_query = (
            f"{query} {dietary_restrictions} for {health_condition}. "
            f"Optimize for {nutritional_goals}."
        )

        query_embedding = model.encode(augmented_query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        context = "\n\n".join(documents) if documents else "No relevant recipes found."
        latency = time.time() - start_time

        # Call OpenRouter
        response = openai.ChatCompletion.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a nutrition expert. Suggest recipes, substitutions, and track goals based on context."},
                {"role": "user",
                 "content": f"Context: {context}\nQuery: {augmented_query}\n"
                            f"Provide: 1. Recommended recipes. "
                            f"2. Substitutions for restrictions/allergies. "
                            f"3. Nutritional analysis. "
                            f"4. Why it matches health condition."}
            ]
        )

        generated = response.choices[0].message["content"]

        # Simple accuracy estimation
        accuracy = 1.0 if documents and any(
            term in context.lower() for term in dietary_restrictions.lower().split()
        ) else 0.5

        return generated, latency, accuracy, metadatas

    except Exception as e:
        print(f"Error in rag_query: {e}")
        raise
