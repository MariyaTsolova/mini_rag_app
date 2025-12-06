import streamlit as st
from sentence_transformers import SentenceTransformer
import psycopg2
import os

# Load secrets from Streamlit
DB_PARAMS = {
    "dbname": st.secrets["DB"]["PGDATABASE"],
    "user": st.secrets["DB"]["PGUSER"],
    "password": st.secrets["DB"]["PGPASSWORD"],
    "host": st.secrets["DB"]["PGHOST"],
    "port": st.secrets["DB"]["PGPORT"],
}

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embedder = load_model()

# Database connection
@st.cache_resource
def connect_db():
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    return conn

conn = connect_db()
cursor = conn.cursor()

st.title("Case Study Semantic Search")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Search"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Embedding & searching..."):
            # Create embedding
            query_embedding = embedder.encode(prompt).flatten().tolist()

            # Query
            similarity_query = """
                SELECT content, meta, embedding <=> %s::vector
                FROM public.case_studies_kb
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
            """

            cursor.execute(similarity_query, (query_embedding, query_embedding))
            results = cursor.fetchall()

        st.subheader("Top Similar Results")
        for idx, (content, meta, distance) in enumerate(results, start=1):
            st.markdown(f"### Result {idx}")
            st.write(content)
            st.write("**Meta:**", meta)
            st.write(f"**Distance:** {distance}")
            st.markdown("---")
