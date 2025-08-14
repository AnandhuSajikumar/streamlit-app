# RecipeRAG: Nutrition RAG System

## Overview
A RAG system for personalized recipe suggestions considering dietary restrictions, health conditions, and nutritional goals. Uses HF embeddings, Chroma vector DB, and OpenAI for generation.

## Setup
1. Clone repo: `git clone <your-repo-url>`
2. Activate venv: `source venv/bin/activate`
3. Install deps: `pip install -r requirements.txt`
4. Add OpenAI key in `utils/rag_utils.py`
5. Download data to `data/recipes.json`
6. Run: `streamlit run main.py`

## Deployment
- Push to GitHub.
- Deploy to Hugging Face Spaces: Create a Space, upload files, set `main.py` as entry.

## Evaluation
- Metrics: Latency (~1-2s), Accuracy (term matching).
- Chunking: Full recipe text as chunks.
- UX: Tabs for navigation.

Public App Link: [Your HF Link Here]

GitHub: [Your Repo Link]