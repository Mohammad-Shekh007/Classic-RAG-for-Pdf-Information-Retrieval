# RAG Prototype: Classic Setup for PDF Documents

## Overview
This project demonstrates a classic **Retrieval-Augmented Generation (RAG)** prototype:
- Documents (PDFs) are loaded from a local directory.
- Embeddings are generated **using OpenAI** (cloud-based).
- GPT-4.1 is used **only for answering queries** based on retrieved text chunks.

> **Important:** Full documents are sent to OpenAI during embedding. Only relevant chunks are sent to GPT-4.1 at query time.

---

## Workflow

### 1. Load Environment
- Load `.env` file containing `OPENAI_API_KEY`.
- Apply `nest_asyncio` for async compatibility in notebooks.

```python
import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# RAG Prototype: Classic Setup for PDF Documents

## Overview
This project demonstrates a classic **Retrieval-Augmented Generation (RAG)** prototype:
- Documents (PDFs) are loaded from a local directory.
- Embeddings are generated **using OpenAI** (cloud-based).
- GPT-4.1 is used **only for answering queries** based on retrieved text chunks.

> **Important:** Full documents are sent to OpenAI during embedding. Only relevant chunks are sent to GPT-4.1 at query time.

---

## Workflow

### 1. Load Environment
- Load `.env` file containing `OPENAI_API_KEY`.
- Apply `nest_asyncio` for async compatibility in notebooks.

```python
import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


2. Load PDF Documents

Use PyMuPDF (fitz) to read PDFs.

Extract text from each page.

Convert each PDF into a Document object for LlamaIndex.

import fitz
from llama_index.core import Document, SimpleDirectoryReader

pdf_dir = r"D:\Your\PDF\Directory"
documents = []

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text().strip()
        if len(text) > 0:
            pages_text.append(text)
    return "\n".join(pages_text)

for file in os.listdir(pdf_dir):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, file)
        content = extract_text_from_pdf(pdf_path)
        documents.append(Document(text=content, metadata={"source": file}))

3. Build Vector Index

VectorStoreIndex converts document text into embeddings via OpenAI API.

Stored locally for retrieval.
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

4. Initialize LLM for Querying

Only the retrieved chunks are sent to GPT-4.1.

LLM generates the answer without seeing the full document.

from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=1000
)

5. Setup Query Engine

Retrieves top-k relevant text chunks.

Passes them to LLM for answer generation.

Streaming and verbose options enabled.

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="compact",
    streaming=True,
    verbose=True
)


6. Query Loop

User inputs a question.

LLM responds based on the retrieved context.

while True:
    q = input("\nEnter your question (or 'exit'): ")
    if q.lower() == "exit":
        break
    response = query_engine.query(q)
    print("\n--- Answer ---")
    print(response)


Key Points / Facts

Embeddings: Cloud-based, OpenAI sees full document text.

Vector Storage: Local, in-memory.

Querying: LLM sees only retrieved chunks, not entire documents.

Purpose: Fast and relevant answers without sending the full document to the model at query time.

# Be mindful of Using Private and Sensitive Information. The document goes to OpenAI for embeddings so use only public data for this setup. #

Next Steps / Future Work

Integrate local embeddings (HuggingFace-based) to fully avoid sending documents to OpenAI.

Implement Knowledge Graph (KG) RAG and Ontology-based RAG for entity-aware reasoning.

Add OCR support to handle scanned PDFs with images.



