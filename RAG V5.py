import os
import fitz  # PyMuPDF
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# --------------------------------
# Async fix
# --------------------------------
nest_asyncio.apply()

# --------------------------------
# Load environment variables
# --------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------
# PDF text loader (text layer only)
# --------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text().strip()
        if len(text) > 0:
            pages_text.append(text)
    return "\n".join(pages_text)

# --------------------------------
# Load PDFs from a directory
# --------------------------------
pdf_dir = r"Your File Path Here"
documents = []

for file in os.listdir(pdf_dir):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, file)
        print(f"Processing: {file}")
        content = extract_text_from_pdf(pdf_path)
        documents.append(Document(text=content, metadata={"source": file}))

print(f"\nLoaded {len(documents)} documents.\n")

# --------------------------------
# Build vector index (local embeddings)
# --------------------------------
index = VectorStoreIndex.from_documents(documents)

# --------------------------------
# OpenAI LLM (Query side only)
# --------------------------------
llm = OpenAI(
    model="gpt-4.1",  # or FLU if available in your account
    temperature=0,
    max_tokens=1000
)

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="compact",
    streaming=True,
    verbose=True
)

# --------------------------------
# Query loop
# --------------------------------
while True:
    q = input("\nEnter your question (or 'exit'): ")
    if q.lower() == "exit":
        break

    response = query_engine.query(q)
    print("\n--- Answer ---")
    print(response)

