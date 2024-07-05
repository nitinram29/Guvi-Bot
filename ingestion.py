from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv

from const import INDEX_NAME

load_dotenv()


class CustomReadTheDocsLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        documents = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        soup = BeautifulSoup(content, "html.parser")
                        body_content = (
                            soup.body.get_text(separator="\n", strip=True)
                            if soup.body
                            else ""
                        )
                        doc = Document(
                            page_content=body_content, metadata={"source": file_path}
                        )
                        documents.append(doc)
        return documents


def ingest_docs() -> None:
    # loader = CustomReadTheDocsLoader("langchain-doc/python.langchain.com")
    loader = CustomReadTheDocsLoader("guvi/www.guvi.in")
    raw_documents = loader.load()
    print(f"raw_documents length : {len(raw_documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50, separators=["\n\n", "\n", "", " "]
    )
    print(f"text_splitter length : {text_splitter}")
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"documents length : {len(documents)}")
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-doc", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = AzureOpenAIEmbeddings(
        deployment="ocean_emb",
        azure_endpoint="https://oceanfreightailabs.openai.azure.com/",
        model="text-embedding-ada-002",
    )
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )


if __name__ == "__main__":
    ingest_docs()
