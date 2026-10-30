# 🎯 AI-Powered Career Guidance Chatbot
Suggests career paths, skills and learning resources from a user's profile using **Retrieval-Augmented Generation (RAG)** and a **Large Language Model**.

## Problem
Students and freshers struggle to choose a career path and know which skills to learn. Generic LLM answers can hallucinate. This project grounds every answer in a curated knowledge base.

## Architecture
```
User Profile + Question
        │
        ▼
 Query builder ──► Sentence-Transformer embeddings (all-MiniLM-L6-v2)
                          │
                          ▼
              ChromaDB vector store  ◄── chunks of data/knowledge/*.md
                          │  top-k similar chunks
                          ▼
        Prompt = System rules + Context + Profile + Question
                          │
                          ▼
                    LLM (Llama 3.3 70B via Groq API)
                          │
                          ▼
  Structured answer: Career paths · Skill gap · 30-60-90 roadmap · Resources · Sources
```

## Tech Stack
Python · Streamlit · ChromaDB · sentence-transformers · Groq API (Llama 3.3 70B) · python-dotenv

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your free GROQ_API_KEY from console.groq.com/keys
python ingest.py                                      # build vector DB
streamlit run app.py
```
Without an API key the app runs in offline mode and shows the retrieved knowledge only.

## Project Structure
```
app.py            Streamlit UI (profile form + chat + retrieved-context viewer)
ingest.py         Builds the vector database
src/rag.py        Chunking, embedding, storage, retrieval
src/llm.py        Prompt construction and LLM call
data/knowledge/   Knowledge base (add more .md files, re-run ingest.py)
```

## Key Features
- Profile-based personalised guidance (education, skills, interests, goal)
- RAG grounding to reduce hallucination, with transparent "Retrieved context" panel
- Structured output: top 3 careers, skill gap, 30-60-90 day plan, resources, sources
- Multi-turn follow-up questions with chat history
- Extensible: just drop new markdown files into `data/knowledge/`

## Future Scope
Resume upload and parsing, live job-market data, multilingual (Hindi) support, hybrid search with re-ranking, evaluation with RAGAS.
