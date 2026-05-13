from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
import time

def get_pinecone_vectorstore(embedding):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "medical-chatbot"

    for attempt in range(3):
        try:
            existing = [i["name"] for i in pc.list_indexes()]
            if index_name not in existing:
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(5)

            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"✅ Pinecone connected! Vectors: {stats.total_vector_count}")
            return PineconeVectorStore(index_name=index_name, embedding=embedding)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(3)
            else:
                raise e