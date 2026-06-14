import os
import shutil
from datetime import datetime

UPLOAD_DIR = "uploaded_files"


def save_uploaded_file(uploaded_file, subfolder=None):
    """Saves a Streamlit uploaded file to local disk. Returns saved path."""

    save_dir = UPLOAD_DIR
    if subfolder:
        save_dir = os.path.join(UPLOAD_DIR, subfolder)

    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    return filepath


def list_saved_files():
    """Returns list of previously uploaded files with metadata."""
    if not os.path.exists(UPLOAD_DIR):
        return []

    files = []
    for fname in sorted(os.listdir(UPLOAD_DIR), reverse=True):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            files.append({
                "name": fname,
                "path": fpath,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).strftime("%Y-%m-%d %H:%M")
            })
    return files
