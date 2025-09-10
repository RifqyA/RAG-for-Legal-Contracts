# RAG Pipeline for Legal and General Document QA

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** refactored from a Jupyter notebook into a Python package. It provides a modular framework for document ingestion, retrieval, and generation, enabling efficient question answering over contracts, reports, or other structured text.

The system:
- Loads and preprocesses documents with **heading-based chunking** and **header/footer cleanup**.  
- Uses **BM25** and **dense embeddings** (HuggingFace, MistralAI) for **hybrid retrieval**.  
- Improves retrieval accuracy with a **Cross-Encoder reranker**.  
- Supports **language detection**, **fuzzy heading matching**, and **intent-aware query handling**.  
- Generates answers with **LLMs** (Mistral or HuggingFace Transformers).  

---

## Document Preprocessing
- **Chunking** based on headings and regex patterns.  
- **Noise removal**: headers, footers, multi-column handling.  
- **Metadata assignment** for traceability across retrieval steps.  

---

## Retrieval and Reranking
- **BM25 Retriever** for keyword relevance.  
- **Dense Vector Retrieval** using HuggingFace and MistralAI embeddings.  
- **Hybrid fusion** of lexical + semantic signals.  
- **Cross-Encoder reranking** (Sentence Transformers) for precision.  

---

## Generation
- Query passed to selected **LLM (Mistral, HuggingFace models)**.  
- Retrieved context is injected for **context-aware responses**.  
- Supports clause/party extraction experiments with **Label Studio**.  

---

## Features
- Hybrid retrieval combining semantic and lexical search.  
- Heading-based document chunking for improved context segmentation.  
- Cross-encoder reranking to boost retrieval accuracy.  
- Intent classification and fuzzy heading matching.  
- Clause/party annotation support (unfinished, experimental).  

---

## Requirements
Key dependencies include:
- `langchain`, `langchain_core`, `langchain_community`, `langchain_huggingface`, `langchain_mistralai`  
- `sentence_transformers`, `transformers`, `torch`  
- `mistralai`, `fitz` (PyMuPDF), `rapidfuzz`, `langid`, `numpy`  

See [requirements.txt](./requirements.txt) for the full list.  

---
