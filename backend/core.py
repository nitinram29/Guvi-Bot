from typing import Any, List, Dict, Tuple

from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from const import INDEX_NAME

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> Any:
    embeddings = AzureOpenAIEmbeddings(
        deployment="ocean_emb",
        azure_endpoint="https://oceanfreightailabs.openai.azure.com/",
        model="text-embedding-ada-002",
    )

    vectorStore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    llm = AzureChatOpenAI(
        model="gpt-4-32k",
        temperature=0,
        azure_endpoint="https://oceanfreightailabs.openai.azure.com/",
        azure_deployment="ocean_gpt4_32k",
    )

    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorStore.as_retriever(),
    #     return_source_documents=True
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorStore.as_retriever(), return_source_documents=True
    )

    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    while True:
        input1 = str(input("Ask question : "))
        if input1 == "exit()":
            break
        print(run_llm(input1))
