import pandas as pd
import pickle
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load the preprocessed data
df = pd.read_csv("Preprocessed_BankFAQs.csv")

# Create a list of Document objects
documents = []
for _, row in df.iterrows():
    content = row["document_content"]
    # Use the 'Class' column as metadata if desired
    metadata = {"class": row["Class"]}
    documents.append(Document(page_content=content, metadata=metadata))

# (Optional) Save the documents list for future use
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

# Initialize Hugging Face Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define a directory to persist your vector store
persist_directory = "chroma_db"

# Create the vector store using Chroma
vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()

print("Vector store built and persisted in:", persist_directory)
