"""
Gradio Frontend — Document RAG Chatbot
"""

import os
import time
import uuid
import requests
import boto3
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Config from environment variables
S3_BUCKET_NAME   = os.environ.get("S3_BUCKET_NAME")
AWS_REGION_NAME  = os.environ.get("AWS_REGION_NAME", "us-east-1")
API_GATEWAY_URL  = os.environ.get("API_GATEWAY_URL")

# S3 client
s3_client = boto3.client("s3", region_name=AWS_REGION_NAME)


def generate_session_id() -> str:
    """
    Generate unique session ID per user upload.
    This keeps each user's FAISS index separate in S3.
    Format: session_<8 random chars>
    """
    return f"session_{uuid.uuid4().hex[:8]}"


def get_filename_without_extension(filepath: str) -> str:
    """Extract filename without extension from full path."""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]


def upload_pdf_to_s3(
    pdf_path: str,
    session_id: str
) -> tuple[str, str]:
    """
    Upload PDF to S3 under pdfs/{session_id}/{filename}.pdf
    This automatically triggers the ingest Lambda via S3 event.

    Returns:
        s3_key:   full S3 path of uploaded PDF
        filename: name without extension (used for querying)
    """
    filename = get_filename_without_extension(pdf_path)
    s3_key = f"pdfs/{session_id}/{filename}.pdf"

    print(f"Uploading PDF to s3://{S3_BUCKET_NAME}/{s3_key}")
    s3_client.upload_file(pdf_path, S3_BUCKET_NAME, s3_key)
    print("Upload complete — ingest Lambda triggered automatically")

    return s3_key, filename


def wait_for_faiss_index(
    session_id: str,
    filename: str,
    max_wait_seconds: int = 120
) -> bool:
    """
    Poll S3 until FAISS index files appear.
    Ingest Lambda creates these after processing the PDF.

    Checks every 5 seconds up to max_wait_seconds.
    Returns True if index is ready, False if timed out.
    """
    faiss_key = f"faiss/{session_id}/{filename}/index.faiss"
    print(f"Waiting for FAISS index at s3://{S3_BUCKET_NAME}/{faiss_key}")

    for attempt in range(max_wait_seconds // 5):
        try:
            s3_client.head_object(
                Bucket=S3_BUCKET_NAME,
                Key=faiss_key
            )
            print(f"FAISS index ready after {attempt * 5} seconds ✅")
            return True
        except s3_client.exceptions.ClientError:
            print(f"Not ready yet... waiting 5 seconds (attempt {attempt + 1})")
            time.sleep(5)

    print("Timed out waiting for FAISS index ❌")
    return False


def query_api_gateway(
    question: str,
    session_id: str,
    filename: str,
    history: list
) -> dict:
    """
    Send question to API Gateway.
    """
    payload = {
        "question":   question,
        "session_id": session_id,
        "filename":   filename,
        "history":    history
    }

    print(f"Calling API Gateway: {API_GATEWAY_URL}")
    response = requests.post(
        API_GATEWAY_URL,
        json=payload,
        timeout=60   # Lambda can take up to 60s on cold start
    )
    response.raise_for_status()
    return response.json()


# ── Gradio Event Handlers ──────────────────────────────────────

def handle_pdf_upload(pdf_file) -> tuple:
    """
    Called when user uploads a PDF.
    """
    if pdf_file is None:
        return None, None, "⚠️ Please upload a PDF file"

    try:
        # Generate unique session for this upload
        session_id = generate_session_id()
        filename = get_filename_without_extension(pdf_file.name)

        # Upload to S3 — triggers ingest Lambda automatically
        yield session_id, filename, "📤 Uploading PDF to S3..."

        upload_pdf_to_s3(pdf_file.name, session_id)

        yield session_id, filename, "⚙️ Processing PDF... this may take 30-60 seconds"

        # Wait for ingest Lambda to finish
        index_ready = wait_for_faiss_index(session_id, filename)

        if index_ready:
            yield session_id, filename, f"✅ '{filename}.pdf' is ready! Ask me anything about it."
        else:
            yield session_id, filename, "❌ Processing timed out. Please try uploading again."

    except Exception as e:
        print(f"Upload error: {str(e)}")
        yield None, None, f"❌ Error: {str(e)}"


def handle_question(
    question: str,
    session_id: str,
    filename: str,
    chat_history: list
) -> tuple:
    """
    Called when user submits a question.
    """
    if not question.strip():
        return chat_history, ""

    if not session_id or not filename:
        chat_history.append((question, "⚠️ Please upload a PDF first!"))
        return chat_history, ""

    try:
        # Call API Gateway
        result = query_api_gateway(
            question,
            session_id,
            filename,
            chat_history
        )

        answer  = result.get("answer", "No answer returned")
        sources = result.get("sources", [])

        # Add sources to answer if available
        if sources:
            sources_text = " | ".join(sources)
            answer = f"{answer}\n\n📄 Sources: {sources_text}"

        chat_history.append((question, answer))
        return chat_history, ""

    except requests.exceptions.Timeout:
        chat_history.append((question, "⚠️ Request timed out. Lambda may be cold starting — try again."))
        return chat_history, ""

    except Exception as e:
        print(f"Query error: {str(e)}")
        chat_history.append((question, f"❌ Error: {str(e)}"))
        return chat_history, ""


# ── Gradio UI Layout ───────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Document RAG Chatbot") as demo:

        gr.Markdown("# 📄 Document Q&A Chatbot")
        gr.Markdown("Upload a PDF and ask questions about it.")

        # Session state — persists across interactions
        session_id_state = gr.State(None)
        filename_state   = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input  = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"]
                )
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload a PDF to get started"
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your document...",
                        scale=4
                    )
                    submit_btn = gr.Button(
                        "Ask",
                        variant="primary",
                        scale=1
                    )

        # ── Event Bindings ──
        pdf_input.change(
            fn=handle_pdf_upload,
            inputs=[pdf_input],
            outputs=[session_id_state, filename_state, status_msg]
        )

        submit_btn.click(
            fn=handle_question,
            inputs=[question_input, session_id_state, filename_state, chatbot],
            outputs=[chatbot, question_input]
        )

        # Allow pressing Enter to submit
        question_input.submit(
            fn=handle_question,
            inputs=[question_input, session_id_state, filename_state, chatbot],
            outputs=[chatbot, question_input]
        )

    return demo


def launch_ui():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
