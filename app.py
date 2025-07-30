import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI 
from utils.embedding import get_embeddings
from utils.pinecone_utils import init_pinecone, get_retriever

# Load environment variables
load_dotenv()
init_pinecone()

# Set up LLM and retriever
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = get_embeddings()
retriever = get_retriever(embeddings)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot – Ask Your Health Questions")

query = st.text_input("Enter your medical question:")

if query:
    response = qa_chain.run(query)
    st.markdown("### 💬 Response")
    st.write(response)
