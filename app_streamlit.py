import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tempfile
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Estudiante AI - RAG", layout="wide", page_icon="📚")

st.title("🤖 Tu Asistente de Algoritmos (RAG)")
st.markdown("Sube tus apuntes y consulta con inteligencia contextual.")

# --- SIDEBAR: CONFIGURACIÓN Y CARGA ---
with st.sidebar:
    st.header("1. Configuración")
    
    # Prioridad: 1. Secrets de Streamlit, 2. Entrada manual
    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        groq_key = st.text_input("Ingresa tu Groq API Key:", type="password")
    else:
        st.success("API Key cargada desde Secrets ✅")

    st.divider()
    st.header("2. Tus Apuntes")
    uploaded_files = st.file_uploader("Sube uno o más PDFs", accept_multiple_files=True, type="pdf")
    
    if st.button("Limpiar historial de chat"):
        st.session_state.messages = []
        st.rerun()

# --- INICIALIZAR MEMORIA DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCIÓN DE PROCESAMIENTO DE PDFS ---
@st.cache_resource
def crear_base_conocimiento(files):
    if not files:
        return None
    
    with st.spinner("Indexando documentos..."):
        all_chunks = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            os.unlink(tmp.name) 
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_db

# --- FLUJO PRINCIPAL ---
if uploaded_files:
    vector_db = crear_base_conocimiento(uploaded_files)
    
    # Mostrar historial de mensajes acumulado
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("¿Qué dice el texto sobre...?"):
        # 1. Mostrar y guardar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generar respuesta
        with st.chat_message("assistant"):
            if not groq_key:
                st.error("Error: Falta la API Key de Groq.")
            else:
                with st.spinner("Consultando contexto y memoria..."):
                    # Búsqueda en vector DB
                    docs_relacionados = vector_db.similarity_search(prompt, k=4)
                    contexto_pdf = "\n\n".join([f"[Pág {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs_relacionados])
                    
                    # Construir historial para el LLM (últimos 4 mensajes para contexto)
                    memoria_reciente = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-5:-1]])
                    
                    # Modelo e Invocación
                    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_key, temperature=0.1)
                    
                    prompt_final = f"""
                    Eres un profesor experto. Responde basándote en el contexto del PDF y el historial de la charla.
                    
                    HISTORIAL DE CONVERSACIÓN:
                    {memoria_reciente}
                    
                    CONTEXTO DEL PDF:
                    {contexto_pdf}
                    
                    PREGUNTA DEL ALUMNO: {prompt}
                    
                    INSTRUCCIÓN: Si la pregunta es ambigua, usa el HISTORIAL para saber de qué tema venimos hablando. 
                    Si la respuesta no está en el CONTEXTO DEL PDF, admítelo.
                    """
                    
                    response = llm.invoke(prompt_final)
                    respuesta_texto = response.content
                    st.markdown(respuesta_texto)
                    
                    # Guardar respuesta en el historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
else:
    st.info("👋 Sube tus archivos PDF para comenzar el chat.")