import os
import uuid

from app.config import MAX_FILE_SIZE, ALLOWED_EXT


def validate_file(file):

    filename = file.name.lower()

    if not any(filename.endswith(e) for e in ALLOWED_EXT):
        raise ValueError("Only PDF allowed")

    if os.path.getsize(file.name) > MAX_FILE_SIZE:
        raise ValueError("File too large")


def secure_name(filename):

    ext = os.path.splitext(filename)[1]

    return f"{uuid.uuid4().hex}{ext}"
