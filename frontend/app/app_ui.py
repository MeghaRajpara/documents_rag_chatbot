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


def upload_pdfs_to_s3(
    pdf_files: list,
    session_id: str
) -> list:
    """
    Upload multiple PDFs to S3 under same session.
    """
    filenames = []
    for pdf_file in pdf_files:
        filename = get_filename_without_extension(pdf_file.name)
        s3_key = f"pdfs/{session_id}/{filename}.pdf"
        print(f"Uploading {filename} to s3://{S3_BUCKET_NAME}/{s3_key}")
        s3_client.upload_file(pdf_file.name, S3_BUCKET_NAME, s3_key)
        filenames.append(filename)
        print(f"Uploaded {filename} ✅")
    return filenames


def wait_for_all_faiss_indexes(
    session_id: str,
    filenames: list,
    max_wait_seconds: int = 180
) -> bool:
    """
    Checks every 5 seconds up to max_wait_seconds.
    Returns True if index is ready, False if timed out.
    """
    print(f"Waiting for {len(filenames)} FAISS indexes...")

    for attempt in range(max_wait_seconds // 5):
        all_ready = True

        for filename in filenames:
            faiss_key = f"faiss/{session_id}/{filename}/index.faiss"
            try:
                s3_client.head_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=faiss_key
                )
                print(f"✅ {filename} index ready")
            except s3_client.exceptions.ClientError:
                print(f"⏳ {filename} not ready yet...")
                all_ready = False
                break

        if all_ready:
            print(f"All indexes ready after {attempt * 5} seconds ✅")
            return True

        time.sleep(5)

    return False


def query_api_gateway(
    question: str,
    session_id: str,
    filenames: list,
    history: list
) -> dict:
    """
    Query across ALL uploaded PDFs and combine answers.
    """
    all_answers = []
    all_sources = []

    for filename in filenames:
        payload = {
            "question":   question,
            "session_id": session_id,
            "filename":   filename,
            "history":    history
        }

        print(f"Querying {filename}...")
        response = requests.post(
            API_GATEWAY_URL,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        answer  = result.get("answer", "")
        sources = result.get("sources", [])

        if answer:
            all_answers.append(f"**{filename}:**\n{answer}")
        if sources:
            all_sources.extend([f"{filename} - {s}" for s in sources])

    return {
        "answer":  "\n\n".join(all_answers),
        "sources": all_sources
    }
# ── Gradio Event Handlers ──────────────────────────────────────

def handle_pdf_upload(pdf_files: list) -> tuple:
    """
    Called when user uploads a PDF.
    """
    if not pdf_files:
        return None, None, "⚠️ Please upload one or more PDF files"

    try:
        session_id = generate_session_id()

        # Get all filenames
        filenames = [
            get_filename_without_extension(f.name)
            for f in pdf_files
        ]
        filenames_str = ", ".join(filenames)

        yield session_id, filenames, f"📤 Uploading {len(pdf_files)} PDF(s) to S3..."

        # Upload all PDFs
        upload_pdfs_to_s3(pdf_files, session_id)

        yield session_id, filenames, f"⚙️ Processing {len(pdf_files)} PDF(s)... this may take 30-90 seconds"

        # Wait for all FAISS indexes
        all_ready = wait_for_all_faiss_indexes(session_id, filenames)

        if all_ready:
            yield session_id, filenames, f"✅ Ready! Loaded: {filenames_str}"
        else:
            yield session_id, filenames, "❌ Processing timed out. Please try again."

    except Exception as e:
        print(f"Upload error: {str(e)}")
        yield None, None, f"❌ Error: {str(e)}"


def handle_question(
    question: str,
    session_id: str,
    filenames: list,
    chat_history: list
) -> tuple:
    if not question.strip():
        return chat_history, ""

    if not session_id or not filenames:
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant",  "content": "⚠️ Please upload a PDF first!"})
        return chat_history, ""

    try:
        # Convert to API format
        history_for_api = []
        for i in range(0, len(chat_history) - 1, 2):
            if i + 1 < len(chat_history):
                history_for_api.append([
                    chat_history[i]["content"],
                    chat_history[i + 1]["content"]
                ])

        result = query_api_gateway(
            question,
            session_id,
            filenames,
            history_for_api
        )

        answer  = result.get("answer", "No answer returned")
        sources = result.get("sources", [])

        if sources:
            sources_text = " | ".join(sources)
            answer = f"{answer}\n\n📄 Sources: {sources_text}"

        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant",  "content": answer})
        return chat_history, ""

    except requests.exceptions.Timeout:
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant",  "content": "⚠️ Request timed out. Try again."})
        return chat_history, ""

    except Exception as e:
        chat_history.append({"role": "user",      "content": question})
        chat_history.append({"role": "assistant",  "content": f"❌ Error: {str(e)}"})
        return chat_history, ""


# ── Gradio UI Layout ───────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Document RAG Chatbot") as demo:

        gr.Markdown("# 📄 Document Q&A Chatbot")
        gr.Markdown("Upload one or more PDFs and ask questions about them.")

        session_id_state = gr.State(None)
        filename_state   = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload PDFs to get started"
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    type="messages"
                )
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your documents...",
                        scale=4
                    )
                    submit_btn = gr.Button(
                        "Ask",
                        variant="primary",
                        scale=1
                    )

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
