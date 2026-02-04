import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key")

# Upload
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXT = [".pdf"]

# Models
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# System Prompt
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant.
You are chatting with a user about Insurel Global Insurance Group.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the documents that might be directly relevant to the user's question:

With this context, please answer the user's question. Be accurate, relevant and complete.
"""
