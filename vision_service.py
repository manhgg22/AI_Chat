import os
import shutil
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Thư mục tạm
TEMP_DIR = "vision_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/analyze")
async def analyze_image(
    prompt: str = Form(""),
    image: UploadFile = File(...)
):
    temp_path = os.path.join(TEMP_DIR, image.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    try:
        base64_img = encode_image(temp_path)
        
        # Ép output JSON cấu trúc
        extract_prompt = f"Trích xuất JSON: category, detected_elements, ocr_content, visual_description, confidence. Context: {prompt}"
        
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": extract_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
