from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

from app.config import (
    CHAT_MODEL,
    OPENAI_API_KEY,
    SYSTEM_PROMPT
)


class RAGChain:

    def __init__(self, retriever):

        self.llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Custom Prompt
        self.prompt = PromptTemplate(
            input_variables=[
                "context",
                "question"
            ],
            template=f"""
                {SYSTEM_PROMPT}

                Context:
                {{context}}

                Question:
                {{question}}

                Answer:
                """
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": self.prompt
            }
        )


    def ask(self, question):

        result = self.chain.invoke({
            "question": question
        })

        return result["answer"]
