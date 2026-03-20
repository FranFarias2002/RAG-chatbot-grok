import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile
import os

# Configuración de página
st.set_page_config(page_title="Mi RAG Chatbot", layout="wide")
st.title("🤖 Estudiante AI: Algoritmos y más")

# Sidebar para configuración y carga
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key", type="password")
    uploaded_files = st.file_uploader("Sube tus PDFs de estudio", accept_multiple_files=True, type="pdf")

# Inicializar memoria en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DE PROCESAMIENTO (Igual a la que ya hiciste) ---
def procesar_pdfs(files):
    all_chunks = []
    for file in files:
        # Guardar temporalmente para que PyPDFLoader lo lea
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_chunks.extend(splitter.split_documents(docs))
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
    return vector_db

# --- INTERFAZ DE CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué quieres saber de tus apuntes?"):
    if not api_key:
        st.error("Por favor, pon tu API Key en la barra lateral.")
    else:
        # Mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            if uploaded_files:
                db = procesar_pdfs(uploaded_files)
                docs = db.similarity_search(prompt, k=3)
                contexto = "\n\n".join([d.page_content for d in docs])
                
                llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=api_key)
                
                # Prompt con memoria básica (enviando los últimos mensajes)
                full_prompt = f"Contexto: {contexto}\n\nPregunta: {prompt}"
                response = llm.invoke(full_prompt)
                full_res = response.content
            else:
                full_res = "Primero sube un PDF para que pueda ayudarte."
            
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})