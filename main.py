import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader # Swapped to single-file loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #2b65ba;
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: bold;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #1a427d;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.title("📚 AI Research Assistant")
st.write("Upload a document and ask questions about it.")

# --- 1. The File Uploader UI ---
uploaded_file = st.file_uploader("Upload your PDF research paper", type=["pdf"])

# We pass the file_path into the cache function so it builds a unique vector store per file
@st.cache_resource(show_spinner="Processing document and building RAG pipeline...")
def setup_pipeline(file_path):
    # Use PyPDFLoader for a single file instead of DirectoryLoader
    Doc_loader = PyPDFLoader(file_path)
    docs = Doc_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunked_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = Chroma.from_documents(
        chunked_docs, embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={'k': 10}
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        temperature=0.2
    )

    model = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an elite, objective, and highly precise AI Research Assistant. Your goal is to help researchers synthesize information, analyze documents, and answer complex queries based strictly on the provided context.

CRITICAL INSTRUCTIONS FOR ACCURACY:
1. Rely ONLY on the clear facts directly mentioned in the Context section below. Do not assume, extrapolate, or bring in outside training data.
2. If the Context does not contain the answer to the User's Question, state clearly and concisely: "I cannot find the answer in the provided research documents." Do not attempt to guess.
3. Keep your tone academic, neutral, and precise.
4. When referencing points from different parts of the context, group them logically.

----------------
CONTEXT:
{context}
----------------"""
        ),
        (
            "human",
            "{prompt}"
        )
    ])

    parser = StrOutputParser()

    parllel_chain = RunnableParallel({
        'context': retriever,
        'prompt': RunnablePassthrough()
    })

    return parllel_chain | prompt | model | parser


# --- 2. Orchestration Logic ---
# Only build the pipeline if a file has actually been uploaded
if uploaded_file is not None:
    
    # Securely save the uploaded file to a temporary directory on the server
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Pass that temporary file path to our pipeline
    main_chain = setup_pipeline(tmp_path)
    
    st.success("Document loaded and chunked successfully! You can now search it.")

    # --- 3. The Search UI ---
    Question = st.text_input("Ask a question based on your documents:")

    if st.button("Search"):
        if Question:
            with st.spinner("Analyzing..."):
                result = main_chain.invoke(Question)
                st.markdown("### Response")
                st.write(result)
        else:
            st.warning("Please enter a question first.")
else:
    # What the user sees before they upload anything
    st.info("👆 Please upload a PDF file to begin.")