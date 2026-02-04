from optparse import Values
import os
import shutil
import gradio as gr

from app.config import UPLOAD_DIR
from app.security import validate_file, secure_name
from app.loader import load_pdf
from app.vector_db import VectorDB
from app.rag_chain import RAGChain


os.makedirs(UPLOAD_DIR, exist_ok=True)


# Global Objects
vector_db = VectorDB()
rag_chain = None


# Upload Handler

def upload_files(files):

    global rag_chain

    if not files:
        return "No files uploaded"

    documents = []

    try:

        for file in files:

            validate_file(file)

            filename = secure_name(file.name)

            save_path = os.path.join(
                UPLOAD_DIR,
                filename
            )

            shutil.copy(file.name, save_path)

            docs = load_pdf(save_path)

            documents.extend(docs)

        if not documents:
            return "No readable content"

        vector_db.build(documents)

        rag_chain = RAGChain(
            vector_db.retriever()
        )

        return "Files uploaded and indexed"

    except Exception as e:

        return f"Upload error: {str(e)}"


# Chat Handler

def chat(message, history):

    global rag_chain

    if not rag_chain:
        return history, "Upload documents first"

    answer = rag_chain.ask(message)

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, ""


# UI Launcher

def launch_ui():
    file_list = ["./example_file/Insurel Claims Processing.pdf", "./example_file/Insurel Commercial Insurance.pdf", \
        "./example_file/Insurel Global Insurance Group.pdf", "./example_file/Insurel Global Insurance-2024.pdf", "./example_file/Insurel WorldCare Plus.pdf"]

    with gr.Blocks(title="Documents Q&A Bot") as app:

        gr.Markdown("# 📄 Documents Q&A Chatbot")

        with gr.Row():

            uploader = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                value=file_list
            )

            upload_btn = gr.Button("Upload")

            status = gr.Textbox(label="Status")

        upload_btn.click(
            upload_files,
            uploader,
            status
        )

        chatbot = gr.Chatbot(
            
        )

        user_box = gr.Textbox(
            placeholder="Ask something..."
        )

        send_btn = gr.Button("Send")

        send_btn.click(
            chat,
            [user_box, chatbot],
            [chatbot, user_box]
        )
        app.launch(theme=gr.themes.Base())
