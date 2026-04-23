"""
Lambda Function: rag-chatbot-query
------------------------------------
Triggered by:  API Gateway POST /query
"""

import json
import os
import boto3
import tempfile

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize AWS clients outside handler
# Best practice — reused across warm Lambda invocations
s3_client = boto3.client("s3")
secrets_client = boto3.client(
    "secretsmanager",
    region_name=os.environ.get("AWS_REGION_NAME", "us-east-1")
)


def get_openai_api_key() -> str:
    """
    Fetch OpenAI API key from AWS Secrets Manager.
    Same pattern as ingest Lambda — consistent across functions.
    """
    secret_name = os.environ["SECRET_NAME"]
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])
    return secret["OPENAI_API_KEY"]


def get_faiss_prefix(session_id: str, filename: str) -> str:
    """
    Build the S3 prefix where the FAISS index lives.
    """
    return f"faiss/{session_id}/{filename}"


def load_faiss_index_from_s3(
    bucket: str,
    session_id: str,
    filename: str,
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Download FAISS index files from S3 and load into memory.
    """
    faiss_prefix = get_faiss_prefix(session_id, filename)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for fname in ["index.faiss", "index.pkl"]:
            s3_key = f"{faiss_prefix}/{fname}"
            local_path = os.path.join(tmp_dir, fname)

            print(f"Downloading s3://{bucket}/{s3_key}")
            s3_client.download_file(bucket, s3_key, local_path)

        # Load FAISS index from local temp directory
        faiss_index = FAISS.load_local(
            tmp_dir,
            embeddings,
            allow_dangerous_deserialization=True  # Safe — we wrote these files ourselves
        )
        print("FAISS index loaded successfully")
        return faiss_index


def build_qa_chain(
    faiss_index: FAISS,
    openai_api_key: str,
    chat_history: list
) -> ConversationalRetrievalChain:
    """
    Build a conversational QA chain using LangChain.
    """
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",      # Cost-effective, fast
        temperature=0             # Deterministic answers for factual QA
    )

    retriever = faiss_index.as_retriever(
        search_kwargs={"k": 4}    # Return top 4 most relevant chunks
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Restore previous chat history into memory
    for human_msg, ai_msg in chat_history:
        memory.chat_memory.add_user_message(human_msg)
        memory.chat_memory.add_ai_message(ai_msg)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,   # Include sources in response
        output_key="answer"
    )

    return chain


def parse_request_body(event: dict) -> dict:
    """
    API Gateway sends body as a JSON string.
    Parse it safely with helpful error messages.
    """
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        raise ValueError("Request body is not valid JSON")

    required_fields = ["question", "session_id", "filename"]
    for field in required_fields:
        if field not in body:
            raise ValueError(f"Missing required field: '{field}'")

    return body


def format_response(status_code: int, body: dict) -> dict:
    """
    Format response for API Gateway.
    CORS headers allow the Gradio frontend to call this API.
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",       # CORS
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body)
    }


def handler(event, context):
    """
    Main Lambda handler — entry point.
    API Gateway calls this on POST /query
    """
    print("Query Lambda triggered")
    print(f"Event: {json.dumps(event)}")

    # Handle CORS preflight request from browser
    if event.get("httpMethod") == "OPTIONS":
        return format_response(200, {"message": "OK"})

    try:
        # Step 1 — Parse request
        body = parse_request_body(event)
        question   = body["question"]
        session_id = body["session_id"]
        filename   = body["filename"]
        history    = body.get("history", [])   # Optional — empty on first question

        print(f"Question: {question}")
        print(f"Session:  {session_id}")
        print(f"File:     {filename}")
        print(f"History:  {len(history)} previous messages")

        # Step 2 — Get config from environment
        bucket_name = os.environ["S3_BUCKET_NAME"]

        # Step 3 — Get OpenAI key from Secrets Manager
        openai_api_key = get_openai_api_key()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        # Step 4 — Load FAISS index from S3
        faiss_index = load_faiss_index_from_s3(
            bucket_name,
            session_id,
            filename,
            embeddings
        )

        # Step 5 — Build QA chain and get answer
        qa_chain = build_qa_chain(faiss_index, openai_api_key, history)
        result = qa_chain.invoke({"question": question})

        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        # Extract page numbers from source documents
        sources = list(set([
            f"Page {doc.metadata.get('page', 'unknown') + 1}"
            for doc in source_docs
        ]))

        print(f"Answer: {answer}")
        print(f"Sources: {sources}")

        return format_response(200, {
            "answer":  answer,
            "sources": sources
        })

    except ValueError as e:
        # Bad request — missing fields etc
        print(f"Bad request: {str(e)}")
        return format_response(400, {"error": str(e)})

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        # Re-raise so Lambda marks as failed
        # and CloudWatch captures full traceback
        raise