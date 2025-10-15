# AI Research Explainer — Retrieval-Augmented Generation (RAG) System

A hands-on project that uses **Retrieval-Augmented Generation (RAG)** to help users read, understand, and query AI research papers using **LangChain**, **OpenAI**, and **Streamlit**.

---

## Overview

This project is part of my ongoing **AI Research Explainer** initiative — a personal learning project where I build a system that:
- Uploads AI research papers (PDFs)
- Splits and embeds them for retrieval
- Uses GPT-4 via OpenAI to answer questions
- Returns grounded, explainable answers from the paper

This marks **Phase 1: Local Prototype** — built and running entirely on my local machine using Anaconda and VS Code.

---

## Tech Stack

| Component | Description |
|------------|-------------|
| **Python (Anaconda)** | Local environment management |
| **VS Code** | Development environment |
| **Streamlit** | Frontend app for user interaction |
| **LangChain** | Manages document loading, chunking & retrieval |
| **ChromaDB** | Local vector database for embeddings |
| **OpenAI API** | Provides GPT-4 powered responses |
| **dotenv** | Securely loads API keys from `.env` file |

---


## Local Setup Instructions (What I Did)

Follow these exact steps to run the project locally 




### 1️⃣ Create a Project Folder
mkdir ai-research-explainer  
cd ai-research-explainer

### 2️⃣ Create a Project Folder  
conda create -n ai-research-explainer python=3.10  
conda activate ai-research-explainer

### 2️⃣ Create and Activate a Conda Environment
conda create -n ai-research-explainer python=3.10  
conda activate ai-research-explainer

### 3️⃣ Install the Required Packages
pip install streamlit langchain langchain-community openai chromadb python-dotenv pypdf

### 4️⃣ Create a .env File in the Project Root
touch .env  
Then open it and add:  
OPENAI_API_KEY=your_openai_api_key_here

### 5️⃣ Run the App
streamlit run app.py


















