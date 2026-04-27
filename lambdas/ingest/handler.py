"""
Lambda Function: rag-chatbot-ingest
------------------------------------
Triggered by: S3 PUT event (pdfs/ prefix)
Purpose:
  1. Download PDF from S3
  2. Extract and chunk text
  3. Create OpenAI embeddings
  4. Save FAISS index back to S3 (faiss/ prefix)
"""

import json
import os
import boto3
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import unquote_plus

# Initialize AWS clients outside handler
# This is best practice — reused across warm Lambda invocations
s3_client = boto3.client("s3")
secrets_client = boto3.client(
    "secretsmanager",
    region_name=os.environ.get("AWS_REGION_NAME", "us-east-1")
)


def get_openai_api_key() -> str:
    """
    Fetch OpenAI API key from AWS Secrets Manager.
    Never hardcode secrets — always fetch at runtime.
    """
    secret_name = os.environ["SECRET_NAME"]
    response = secrets_client.get_secret_value(SecretId=secret_name)
    secret = json.loads(response["SecretString"])
    return secret["OPENAI_API_KEY"]


def get_faiss_s3_prefix(pdf_key: str) -> str:
    """
    Convert PDF S3 key to FAISS index prefix.

    Example:
        pdfs/session123/report.pdf
        → faiss/session123/report
    """
    without_prefix = pdf_key.replace("pdfs/", "", 1)
    without_extension = without_prefix.rsplit(".", 1)[0]
    return f"faiss/{without_extension}"


def download_pdf_from_s3(bucket: str, key: str) -> str:
    """
    Download PDF from S3 to a local temp file.
    Returns the local file path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    s3_client.download_fileobj(bucket, key, tmp)
    tmp.close()
    print(f"Downloaded s3://{bucket}/{key} → {tmp.name}")
    return tmp.name


def extract_and_chunk(pdf_path: str) -> list:
    """
    Load PDF and split into chunks for embedding.

    chunk_size=1000    → each chunk is ~1000 characters
    chunk_overlap=200  → chunks overlap by 200 chars
                         so context isn't lost at boundaries
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Extracted {len(documents)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_and_save_faiss_index(
    chunks: list,
    embeddings: OpenAIEmbeddings,
    bucket: str,
    faiss_prefix: str
) -> None:
    """
    Build FAISS index from chunks and upload to S3.

    FAISS saves two files:
      index.faiss  → the actual vector index
      index.pkl    → metadata (document content, sources)
    """
    print("Building FAISS index...")
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # Save index to temp directory then upload to S3
    with tempfile.TemporaryDirectory() as tmp_dir:
        faiss_index.save_local(tmp_dir)
        print(f"FAISS index saved locally to {tmp_dir}")

        for filename in ["index.faiss", "index.pkl"]:
            local_path = os.path.join(tmp_dir, filename)
            s3_key = f"{faiss_prefix}/{filename}"
            s3_client.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded → s3://{bucket}/{s3_key}")


def handler(event, context):
    """
    Main Lambda handler — entry point.
    AWS calls this function when S3 triggers the Lambda.
    """
    print("Ingest Lambda triggered")
    print(f"Event: {json.dumps(event)}")

    # Extract bucket and key from S3 event
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    pdf_key = unquote_plus(record["s3"]["object"]["key"])

    print(f"Processing: s3://{bucket}/{pdf_key}")

    # Safety check — only process PDFs
    if not pdf_key.lower().endswith(".pdf"):
        print(f"Skipping non-PDF file: {pdf_key}")
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Skipped — not a PDF"})
        }

    # Read config from environment variables
    bucket_name = os.environ["S3_BUCKET_NAME"]

    try:
        # Step 1 — Get OpenAI key from Secrets Manager
        openai_api_key = get_openai_api_key()
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        # Step 2 — Download PDF from S3
        pdf_path = download_pdf_from_s3(bucket, pdf_key)

        # Step 3 — Extract text and chunk it
        chunks = extract_and_chunk(pdf_path)

        # Step 4 — Build FAISS index and save to S3
        faiss_prefix = get_faiss_s3_prefix(pdf_key)
        build_and_save_faiss_index(
            chunks, embeddings, bucket_name, faiss_prefix
        )

        # Step 5 — Clean up temp PDF file
        os.unlink(pdf_path)
        print("Temp PDF cleaned up")

        result = {
            "message": "PDF processed successfully",
            "pdf_key": pdf_key,
            "faiss_prefix": faiss_prefix
        }
        print(f"Success: {json.dumps(result)}")
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Re-raise so Lambda marks invocation as failed
        # and CloudWatch captures the full error
        raise
