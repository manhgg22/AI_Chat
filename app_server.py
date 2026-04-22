from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
from vision_tool import GroqVisionTool
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Ensure temp directory exists
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize tool
try:
    vision_tool = GroqVisionTool()
except Exception as e:
    print(f"Warning: Tool initialization failed: {e}")
    vision_tool = None

@app.get("/")
async def get_root():
    return {"status": "Vision Sub-agent is running", "usage": "POST to /analyze"}

@app.post("/analyze")
async def analyze(
    prompt: str = Form("What's in this image?"),
    image: UploadFile = File(...)
):
    print(f"[VISION SERVICE] Received analysis request: '{prompt}'")
    if not vision_tool:
        print("[VISION SERVICE] Error: Tool not initialized.")
        raise HTTPException(status_code=500, detail="Groq API Key not configured correctly.")

    # Save uploaded file temporarily
    file_path = os.path.join(TEMP_DIR, image.filename)
    print(f"[VISION SERVICE] Processing image: {image.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        # Call the vision tool
        result = vision_tool.analyze_image(prompt, file_path)
        print("[VISION SERVICE] Analysis complete.")
        return {"result": result}
    except Exception as e:
        print(f"[VISION SERVICE] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

