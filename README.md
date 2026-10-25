<div align="center">

# Hybrid Graph-RAG Hotel & Visa AI Assistant + XAI Analytics 🏨🧠

### **Neo4j Knowledge Graph · ChromaDB / FAISS Vector Retrieval · Multi-LLM Orchestration · SHAP Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Graph_RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB & FAISS](https://img.shields.io/badge/Vector_DB-ChromaDB_%2B_FAISS-FF6B6B?style=for-the-badge)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

*An end-to-end hospitality intelligence system combining relational data engineering, Feed-Forward Neural Networks (FFNN) with **SHAP Explainable AI**, and a hybrid **Neo4j Knowledge Graph + Vector RAG** conversational assistant.*

</div>

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                  Streamlit Conversational UI (`MS3.py`)                 │
│   • Hybrid Retrieval Mode Selector (Keyword / Vector / Hybrid KG+RAG)   │
│   • Real-Time Token Telemetry, Latency Metrics & <thinking> Trace UI    │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │                                 │
                    ▼                                 ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────┐
│  Hybrid Graph-RAG Engine              │ │  Visa & Policy PDF RAG        │
│  (`aclmilestone3.py`)                 │ │  (LangChain + FAISS + HF)     │
│  • LLM Intent & Entity Extractor      │ │  • Recursive Text Chunking    │
│  • Neo4j Cypher Traversal             │ │  • HuggingFace Embeddings     │
│  • ChromaDB Semantic Vector Search    │ └───────────────┬───────────────┘
└───────────────────┬───────────────────┘                 │
                    └─────────────────┬───────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                Multi-LLM Generation & Reasoning Layer                   │
│       Google Gemini / Gemma-3-27B · OpenRouter · HuggingFace Hub        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Engineering Milestones

### 1. Relational Data Engineering & Predictive ML with XAI (`acl-milestone1-team57-finalfinalfinal.ipynb`)
- **Hospitality Data Pipeline:** Ingests and normalizes multi-table datasets (`hotels.csv`, `users.csv`, `reviews.csv`, `booking_db.sqlite`) to answer complex demographic queries (e.g., optimal destinations by traveller persona and value-for-money rankings across age cohorts).
- **Data Leakage Diagnosis & Imbalance Mitigation:** Identifies and eliminates target leakage from hotel identifier features and balances skewed rating distributions using **Oversampling and Undersampling**.
- **Shallow Feed-Forward Neural Network (FFNN) + SHAP Explainability:** Trains an FFNN rating predictor and applies **SHAP (SHapley Additive exPlanations)** to quantify feature attribution across cleanliness, comfort, staff, and location scores.

### 2. Hybrid Neo4j Knowledge Graph + ChromaDB Vector Retrieval (`aclmilestone3.py`)
- **LLM Entity & Intent Classification:** Parses natural-language travel queries into structured intents and entity filters verified against the live graph schema.
- **Graph + Vector Fusion:** Combines **Neo4j Cypher** multi-hop relationship queries (`Traveller -> wrote -> Review -> stayed_at -> Hotel -> located_in -> City/Country`) with **ChromaDB** semantic review embeddings (`sentence-transformers`).

### 3. Interactive Multi-Model Streamlit Assistant (`MS3.py`)
- Conversational UI supporting **Keyword**, **Vector**, and **Hybrid Graph-RAG** retrieval modes, PDF visa policy Q&A via **FAISS**, `<thinking>` chain-of-thought collapsible inspection, and live token/latency benchmarking.

---

## 📂 Repository Structure

```text
├── MS3.py                                       # Streamlit conversational UI, PDF FAISS RAG & token telemetry
├── aclmilestone3.py                             # Neo4j Knowledge Graph, ChromaDB vector search & LLM router
├── acl-milestone1-team57-finalfinalfinal.ipynb  # Data engineering, FFNN training, leakage analysis & SHAP XAI
├── booking_db.sqlite                            # Normalized relational SQLite hospitality database
├── hotels.csv                                   # Hotel entity dataset
├── users.csv                                    # Traveller demographic dataset
└── reviews.csv                                  # Multi-dimensional hotel review dataset
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install streamlit streamlit-chat langchain langchain-community langchain-huggingface \
            chromadb faiss-cpu neo4j sentence-transformers google-generativeai pandas numpy shap
```

### 2. Configure Environment (`.env`)
```env
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Launch the Streamlit Assistant
```bash
streamlit run MS3.py
```
