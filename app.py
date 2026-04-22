import os
import json
import base64
import re
import shutil
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from groq import Groq
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==========================================
load_dotenv()
app = FastAPI()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

class UnifiedVisionAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.v_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.c_model = os.getenv("MODEL_ID", "qwen/qwen3-32b")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def chat_logic(self, user_query: str, image_path: Optional[str] = None) -> str:
        """
        Logic ổn định: Phân tích -> Hậu xử lý -> Kết quả cuối cùng
        """
        vision_data = None
        if image_path:
            base64_image = self._encode_image(image_path)
            extract_prompt = f"Phân tích ảnh và trích xuất JSON: category, detected_elements, ocr_content, visual_description, confidence. Focus: {user_query}"
            
            try:
                v_res = self.client.chat.completions.create(
                    model=self.v_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extract_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                vision_data = json.loads(v_res.choices[0].message.content)
            except Exception as e:
                print(f"Error Vision: {e}")

        # System prompt nghiêm ngặt
        system_instruction = f"""
        Bạn là Qwen - Chuyên gia trợ lý ảo thông minh. 
        NHIỆM VỤ: Trả lời câu hỏi người dùng dựa trên dữ liệu hình ảnh (nếu có).
        QUY TẮC BẮT BUỘC:
        1. CHỈ sử dụng Tiếng Việt 100%. Không xen kẽ tiếng Anh.
        2. Tuyệt đối KHÔNG hiển thị các khối suy nghĩ <think>.
        3. Sử dụng Markdown để trình bày kết quả thật đẹp và chuyên nghiệp.
        """
        
        messages = [{"role": "system", "content": system_instruction}]
        if vision_data:
            messages.append({"role": "user", "content": f"Dữ liệu Vision chuyên sâu: {json.dumps(vision_data, ensure_ascii=False)}\nCâu hỏi người dùng: {user_query}"})
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            completion = self.client.chat.completions.create(
                model=self.c_model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=2048
            )
            
            final_text = completion.choices[0].message.content
            # Lọc sạch <think> tags
            final_text = re.sub(r'<think>.*?</think>', '', final_text, flags=re.DOTALL).strip()
            
            # Thêm Footer Độ tin cậy nếu có phân tích ảnh
            if vision_data:
                conf = int(vision_data.get('confidence', 0) * 100)
                final_text += f"\n\n---\n🎯 **Độ tin cậy hệ thống: {conf}%**"
                if conf < 70:
                    final_text += f"\n⚠️ *Lưu ý: {vision_data.get('warning', 'Kết quả này có thể cần kiểm chứng thêm.')}*"
            
            return final_text

        except Exception as e:
            return f"❌ Lỗi hệ thống: {str(e)}"

agent = UnifiedVisionAgent()

# ==========================================
# 3. GIAO DIỆN (STABLE MODE)
# ==========================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>ANTIGRAVITY IQ | STABLE</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root { --primary: #8b5cf6; --bg: #030712; --surface: rgba(17, 24, 39, 0.85); --border: rgba(255, 255, 255, 0.1); }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: radial-gradient(circle at top left, #1e1b4b, #030712); color: #f3f4f6; height: 100vh; display: flex; justify-content: center; align-items: center; }
        .app-container { width: 90%; max-width: 1000px; height: 85vh; background: var(--surface); backdrop-filter: blur(25px); border: 1px solid var(--border); border-radius: 28px; display: flex; flex-direction: column; box-shadow: 0 25px 50px rgba(0,0,0,0.5); }
        .header { padding: 25px; text-align: center; border-bottom: 1px solid var(--border); }
        .header h1 { font-size: 1.8rem; background: linear-gradient(to right, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
        .chat-box { flex: 1; padding: 30px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }
        .message { max-width: 80%; padding: 16px 22px; border-radius: 20px; font-size: 0.95rem; line-height: 1.6; animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user-message { align-self: flex-end; background: #6366f1; border-bottom-right-radius: 4px; }
        .ai-message { align-self: flex-start; background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
        .input-area { padding: 25px 35px; border-top: 1px solid var(--border); background: rgba(0,0,0,0.2); }
        .form-container { display: flex; gap: 15px; align-items: center; }
        .upload-trigger { width: 50px; height: 50px; background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 12px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 20px; }
        input[type="text"] { flex: 1; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: white; padding: 15px 20px; border-radius: 14px; outline: none; }
        .send-btn { background: white; color: black; border: none; padding: 0 25px; height: 50px; border-radius: 12px; font-weight: 700; cursor: pointer; transition: transform 0.2s; }
        .send-btn:hover { transform: scale(1.05); }
        #preview { display: none; width: 100px; height: 70px; object-fit: cover; border-radius: 10px; margin-bottom: 10px; border: 2px solid var(--primary); }
        .loading { display:none; color: var(--primary); font-size: 0.8rem; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header"><h1>ANTIGRAVITY IQ | STABLE</h1></div>
        <div class="chat-box" id="chatBox"><div class="message ai-message">Xin chào! Hệ thống Stable IQ đã sẵn sàng để phục vụ bạn.</div></div>
        <div class="input-area">
            <img id="preview" src="">
            <div id="loader" class="loading">Đang phân tích xử lý...</div>
            <form id="chatForm" class="form-container">
                <input type="file" id="imageInput" accept="image/*" style="display:none" onchange="previewFile()">
                <div class="upload-trigger" onclick="document.getElementById('imageInput').click()">📷</div>
                <input type="text" id="messageInput" placeholder="Nhập câu hỏi tại đây..." autocomplete="off">
                <button type="submit" class="send-btn" id="sendBtn">GỬI</button>
            </form>
        </div>
    </div>
    <script>
        const chatBox = document.getElementById('chatBox');
        const chatForm = document.getElementById('chatForm');
        const messageInput = document.getElementById('messageInput');
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const loader = document.getElementById('loader');

        function previewFile() { const file = imageInput.files[0]; if(file) { const reader = new FileReader(); reader.onload=(e)=>{preview.src=e.target.result; preview.style.display='block';}; reader.readAsDataURL(file); } }

        chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const msg = messageInput.value;
            const file = imageInput.files[0];
            if(!msg && !file) return;

            // Render User message
            const userDiv = document.createElement('div'); userDiv.className='message user-message'; userDiv.innerText=msg || "Phân tích ảnh"; chatBox.appendChild(userDiv);
            
            // Clear input & Reset preview
            messageInput.value=''; preview.style.display='none'; loader.style.display='block';
            document.getElementById('sendBtn').disabled = true;

            const fd = new FormData();
            fd.append('message', msg); if(file) fd.append('image', file);
            imageInput.value='';

            try {
                const res = await fetch('/chat', { method: 'POST', body: fd });
                const data = await res.json();
                const aiDiv = document.createElement('div'); aiDiv.className='message ai-message'; 
                aiDiv.innerHTML = marked.parse(data.response);
                chatBox.appendChild(aiDiv);
            } catch (err) {
                alert("Lỗi: " + err.message);
            } finally {
                loader.style.display='none';
                document.getElementById('sendBtn').disabled = false;
                chatBox.scrollTo(0, chatBox.scrollHeight);
            }
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index(): return HTML_CONTENT

@app.post("/chat")
async def chat_endpoint(message: str = Form(""), image: UploadFile = File(None)):
    temp_path = None
    if image and image.filename:
        temp_path = os.path.join(TEMP_DIR, image.filename)
        with open(temp_path, "wb") as f: shutil.copyfileobj(image.file, f)
    
    try:
        response_text = agent.chat_logic(message, temp_path)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
