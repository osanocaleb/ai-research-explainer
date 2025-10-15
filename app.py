
from dotenv import load_dotenv
load_dotenv() 

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="AI Research Explainer", page_icon="📚")

st.title("AI Research Explainer (RAG Demo)")
st.markdown("""
Upload a **research paper PDF** and ask questions about it.
This demo uses **OpenAI (GPT-4)** for reasoning and **ChromaDB** for retrieval.
""")


uploaded_file = st.file_uploader("📄 Upload a research paper (PDF):", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Save file locally
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Load PDF 
    loader = PyPDFLoader("uploaded_doc.pdf")
    pages = loader.load()
    st.write(f"📘 Total Pages: {len(pages)}")

    # Step 2: Split Text into Chunks
    st.write("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(pages)
    st.write(f"🔹 Total Chunks Created: {len(docs)}")

    # Step 3: Create Embeddings + Vector Store 
    st.write("🧠 Creating embeddings and storing in ChromaDB...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Step 4: Create Retriever + LLM Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Use your OpenAI API key from .env
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        st.stop()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Step 5: Question Input 
    st.markdown("### 🔍 Ask a question about the paper")
    user_query = st.text_input("Type your question below:")

    if user_query:
        with st.spinner("🤔 Thinking..."):
            result = qa_chain.invoke({"query": user_query})
            st.write("### Answer:")
            st.write(result["result"])

            # Step 6: Show Sources
            with st.expander("View Source Texts"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source {i} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.write(doc.page_content[:500])
else:
    st.info("👆 Upload a research paper to start exploring it with AI.")

