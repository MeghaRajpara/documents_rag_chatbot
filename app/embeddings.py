import openai
import numpy as np

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY


def create_embedding(text):

    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return np.array(
        response.data[0].embedding,
        dtype="float32"
    )
