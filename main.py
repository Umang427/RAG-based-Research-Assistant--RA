from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

### ---MAIN APP---

load_dotenv()

#--- doc preaparing area ---
Doc_loader= DirectoryLoader(
    "docs",
    glob="**/*.*",
    loader_cls=UnstructuredFileLoader
)
docs=Doc_loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=200
)
chunked_docs = splitter.split_documents(docs)

#--- using a free hugging face Api for embedding ---
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

#--- vector store ---
vector_store=Chroma.from_documents(
    chunked_docs,embeddings
)

#--- retrival ---
retrival=vector_store.as_retriever(
    search_type="similarity",search_kwargs={'k':10}
)
Question=input("Tell me the question")


#--- llm and prompting and parser---
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    temperature=0.2
)

model=ChatHuggingFace(llm=llm)

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

parser=StrOutputParser()

parllel_chain = RunnableParallel({
    'context': retrival,
    'prompt':RunnablePassthrough()
})

main_chain= parllel_chain | prompt | model | parser

result = main_chain.invoke(Question)
print('<----AI Response---->')
print(result)