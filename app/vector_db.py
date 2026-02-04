from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import OPENAI_API_KEY, EMBED_MODEL


class VectorDB:

    def __init__(self):

        self.db = None

        self.embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY
        )


    def build(self, documents):

        self.db = FAISS.from_documents(
            documents,
            self.embeddings
        )


    def retriever(self):

        if not self.db:
            return None

        return self.db.as_retriever(
            search_kwargs={"k": 4}
        )
