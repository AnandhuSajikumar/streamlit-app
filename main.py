import streamlit as st
from utils.rag_utils import load_data, ingest_to_chroma, rag_query
import matplotlib.pyplot as plt
import pandas as pd

# Custom black theme for professional look
st.markdown("""
    <style>
    .main {background-color: #1a1a1a; color: #ffffff;}
    .stButton>button {background-color: #2ecc71; color: white; border-radius: 5px;}
    .stTextInput>div>div>input {background-color: #333333; color: #ffffff; border-radius: 5px; border: 1px solid #555555;}
    .stSelectbox>div>div>select {background-color: #333333; color: #ffffff; border-radius: 5px; border: 1px solid #555555;}
    .stTabs [data-baseweb="tab-list"] {background-color: #2a2a2a;}
    .stTabs [data-baseweb="tab"] {background-color: #2a2a2a; color: #ffffff;}
    .stTabs [data-baseweb="tab"]:hover {background-color: #3a3a3a;}
    .stTabs [data-baseweb="tab--selected"] {background-color: #3a3a3a; color: #2ecc71; border-bottom: 2px solid #2ecc71;}
    .stHeader {color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

st.title("RecipeRAG: Personalized Nutrition Advisor")
st.subheader("Get tailored recipes based on your diet, health, and goals")

# Rest of your code remains the same...

# Sidebar for inputs
with st.sidebar:
    st.header("User Preferences")
    dietary_restrictions = st.multiselect("Dietary Restrictions/Allergies", ["Vegan", "Gluten-Free", "Nut-Free", "Low-Carb", "Dairy-Free"])
    health_condition = st.selectbox("Health Condition", ["None", "Diabetes", "Heart Health", "Weight Loss", "Hypertension"])
    nutritional_goals = st.text_input("Nutritional Goals (e.g., high protein, low calories)")
    cuisine = st.text_input("Preferred Cuisine (e.g., Indian, Mediterranean)")

# Tabs for UX
tab1, tab2, tab3 = st.tabs(["Suggest Recipes", "View Data", "Metrics"])

with tab1:
    query = st.text_input("What meal are you looking for? (e.g., quick dinner)")
    if st.button("Get Suggestions"):
        with st.spinner("Processing..."):
            try:
                df = load_data("data/recipes.json")
                collection, model = ingest_to_chroma(df)
                restrictions_str = " ".join([f"-{r.lower()}" for r in dietary_restrictions])
                response, latency, accuracy, metadatas = rag_query(query, collection, model, restrictions_str, health_condition, nutritional_goals)
                
                st.success("Recommendations:")
                st.write(response)
                
                if metadatas:
                    sample_nut = {"Calories": 500, "Protein": 30, "Carbs": 40, "Fat": 20}
                    fig, ax = plt.subplots()
                    pd.Series(sample_nut).plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title("Sample Nutrition Breakdown")
                    st.pyplot(fig)
                
                st.info("Ingredient Substitutions: e.g., Use tofu instead of meat for vegan.")
                
                # Display metrics within the suggestion block
                st.write(f"Retrieval Latency: {latency:.2f} seconds")
                st.write(f"Basic Accuracy: {accuracy * 100}% (term match in retrieved docs)")
            except Exception as e:
                st.error(f"Error processing request: {e}")

with tab2:
    st.header("Sample Recipes Data")
    try:
        df = load_data("data/recipes.json")
        st.dataframe(df.head(10)[['name', 'nutrition_str']])
    except Exception as e:
        st.error(f"Error loading data: {e}")

with tab3:
    st.header("System Metrics")
    st.write("Metrics are displayed under 'Suggest Recipes' after a query.")

# Footer
st.markdown("---\nBuilt with Streamlit | © 2025 RecipeRAG")