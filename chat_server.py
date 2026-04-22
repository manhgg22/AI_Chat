from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import shutil
import os
from chat_agent import ChatAgent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration
VISION_SERVICE_URL = os.environ.get("VISION_SERVICE_URL", "http://localhost:8080/analyze")
MODEL_ID = os.environ.get("MODEL_ID", "qwen/qwen3-32b")
TEMP_DIR = "temp_chat_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Agent
try:
    chat_agent = ChatAgent(model=MODEL_ID, vision_service_url=VISION_SERVICE_URL)
except Exception as e:
    print(f"Warning: Chat Agent initialization failed: {e}")
    chat_agent = None

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("chat_ui.html", "r", encoding="utf-8") as f:
        return f.read()

from typing import List

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    images: List[UploadFile] = File(None)
):
    print(f"\n[SERVER] New request: '{message}'")
    if not chat_agent:
        print("[SERVER] Error: Chat Agent not initialized.")
        raise HTTPException(status_code=500, detail="Chat Agent not initialized.")

    file_paths = []
    try:
        if images:
            for image in images:
                if image.filename:
                    print(f"[SERVER] Image uploaded: {image.filename}")
                    file_path = os.path.join(TEMP_DIR, f"{len(file_paths)}_{image.filename}")
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(image.file, buffer)
                    file_paths.append(file_path)

        response = chat_agent.chat(message, file_paths if file_paths else None)
        print("[SERVER] Request processed successfully.")
        return {"response": response}
    except Exception as e:
        print(f"[SERVER] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
