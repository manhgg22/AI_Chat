import os
import json
import requests
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Cấu hình kết nối sang dịch vụ Vision
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://vision_service:8080/analyze")

class ChatAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("MODEL_ID", "qwen/qwen3-32b")

    def chat_logic(self, user_query: str, image_file: Optional[UploadFile] = None) -> str:
        vision_data = None
        
        # 1. Nếu có ảnh, gọi sang dịch vụ Vision (8080)
        if image_file:
            print(f"[AGENT] Gửi ảnh sang Vision Service: {VISION_SERVICE_URL}")
            try:
                files = {"image": (image_file.filename, image_file.file, image_file.content_type)}
                data = {"prompt": user_query}
                response = requests.post(VISION_SERVICE_URL, files=files, data=data)
                if response.ok:
                    vision_data = response.json()
                else:
                    print(f"[AGENT] Vision Service Error: {response.text}")
            except Exception as e:
                print(f"[AGENT] Connection Error: {e}")

        # 2. Xử lý Trí tuệ (Brain Layer)
        system_instruction = "Bạn là Qwen Assistant. Trả lời TIẾNG VIỆT 100%. Không Anh, không <think>. Trình bày Markdown đẹp."
        messages = [{"role": "system", "content": system_instruction}]
        
        if vision_data:
            messages.append({"role": "user", "content": f"Dữ liệu ảnh: {json.dumps(vision_data, ensure_ascii=False)}\nCâu hỏi: {user_query}"})
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            completion = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.6)
            txt = completion.choices[0].message.content
            txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.DOTALL).strip()
            
            if vision_data:
                conf = int(vision_data.get('confidence', 0) * 100)
                txt += f"\n\n---\n🎯 **Độ tin cậy: {conf}%**"
            return txt
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"

agent = ChatAgent()

# ==========================================
# GIAO DIỆN (GIỮ NGUYÊN STYLE CHATGPT)
# ==========================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Antigravity Chat Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root { --sidebar-bg:#000000; --main-bg:#0a0a0a; --border:#262626; --text:#ececec; --text-dim:#9b9b9b; --primary:#3b82f6; }
        * { margin:0; padding:0; box-sizing:border-box; font-family:'Plus Jakarta Sans', sans-serif; }
        body { background:var(--main-bg); color:var(--text); height:100vh; display:flex; overflow:hidden; }
        #sidebar { width:260px; height:100%; background:var(--sidebar-bg); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:12px; }
        .new-chat-btn { border:1px solid var(--border); padding:12px; border-radius:8px; cursor:pointer; margin-bottom:20px; text-align:center; font-weight:500; display:flex; align-items:center; gap:8px; justify-content:center; }
        .new-chat-btn:hover { background:#171717; }
        .history-list { flex:1; overflow-y:auto; }
        .history-item { padding:10px 12px; border-radius:8px; cursor:pointer; font-size:0.85rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:4px; color:var(--text-dim); }
        .history-item:hover { background:#171717; color:white; }
        #main { flex:1; display:flex; flex-direction:column; position:relative; }
        .header { padding:16px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
        .header h2 { font-size:0.9rem; font-weight:600; color:var(--text-dim); }
        #chat-container { flex:1; overflow-y:auto; padding:40px 15%; display:flex; flex-direction:column; gap:32px; }
        .message-row { display:flex; gap:20px; animation:fadeIn 0.3s ease-out; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
        .avatar { width:32px; height:32px; border-radius:6px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:12px; }
        .user-avatar { background:#ef4444; color:white; }
        .ai-avatar { background:#10b981; color:white; }
        .message-content { flex:1; line-height:1.6; font-size:1.05rem; }
        .input-wrapper { padding:20px 15%; background:var(--main-bg); }
        .input-container { background:#171717; border:1px solid var(--border); border-radius:12px; padding:12px 16px; display:flex; align-items:flex-end; gap:12px; box-shadow:0 4px 12px rgba(0,0,0,0.4); }
        textarea { flex:1; background:transparent; border:none; color:white; outline:none; resize:none; padding:4px; font-size:1rem; max-height:200px; }
        .icon-btn { cursor:pointer; color:var(--text-dim); transition:0.2s; padding:4px; }
        .icon-btn:hover { color:white; }
        #preview-box { display:none; margin-bottom:10px; width:60px; height:60px; position:relative; }
        #preview-box img { width:100%; height:100%; object-fit:cover; border-radius:6px; }
        .remove-img { position:absolute; top:-8px; right:-8px; background:white; color:black; border-radius:50%; width:16px; height:16px; font-size:10px; display:flex; align-items:center; justify-content:center; cursor:pointer; }
        .loading-dots { display:none; font-size:12px; color:var(--text-dim); margin-bottom:8px; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="new-chat-btn" onclick="startNewChat()">+ New Chat</div>
        <div class="history-list" id="historyList"></div>
    </div>
    <div id="main">
        <div class="header"><h2>Antigravity Agent</h2><div style="font-size:11px;color:#10b981">● Connected</div></div>
        <div id="chat-container"></div>
        <div class="input-wrapper">
            <div id="preview-box"><img id="preview-img"><div class="remove-img" onclick="clearImage()">×</div></div>
            <div class="loading-dots" id="loader">Đang xử lý qua Vision Service...</div>
            <div class="input-container">
                <label class="icon-btn"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg><input type="file" id="imageInput" style="display:none" onchange="previewFile()"></label>
                <textarea id="messageInput" placeholder="Hỏi tôi về bất cứ điều gì..." rows="1" oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px'"></textarea>
                <button class="icon-btn" style="border:none;background:transparent" onclick="sendMessage()"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polyline points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
            </div>
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('messageInput');
        const imageInput = document.getElementById('imageInput');
        const loader = document.getElementById('loader');
        let currentMessages = [];

        function previewFile() { const file=imageInput.files[0]; if(file) { const r=new FileReader(); r.onload=(e)=>{document.getElementById('preview-img').src=e.target.result; document.getElementById('preview-box').style.display='block';}; r.readAsDataURL(file); } }
        function clearImage() { imageInput.value=''; document.getElementById('preview-box').style.display='none'; }
        
        function addMessageRow(text, role) {
            const row=document.createElement('div'); row.className='message-row';
            const ava=document.createElement('div'); ava.className=`avatar ${role==='ai'?'ai-avatar':'user-avatar'}`; ava.innerText=role==='ai'?'AI':'U';
            const cont=document.createElement('div'); cont.className='message-content';
            if(role==='ai') cont.innerHTML=marked.parse(text); else cont.innerText=text;
            row.appendChild(ava); row.appendChild(cont); chatContainer.appendChild(row); chatContainer.scrollTo(0, chatContainer.scrollHeight);
        }

        async function sendMessage() {
            const msg=messageInput.value.trim(); const file=imageInput.files[0];
            if(!msg && !file) return;
            addMessageRow(msg || "[Ảnh]", 'user');
            messageInput.value=''; messageInput.style.height='auto'; clearImage(); loader.style.display='block';
            const fd=new FormData(); fd.append('message', msg); if(file) fd.append('image', file);
            try {
                const res=await fetch('/chat', {method:'POST', body:fd});
                const data=await res.json(); addMessageRow(data.response, 'ai');
            } catch(e) { addMessageRow("❌ Lỗi: "+e.message, 'ai'); } finally { loader.style.display='none'; }
        }

        function startNewChat() { chatContainer.innerHTML=''; addMessageRow("Hệ thống **Multi-Service Vision** đã sẵn sàng.", 'ai'); }
        startNewChat();
        messageInput.addEventListener('keydown', (e)=>{if(e.key==='Enter' && !e.shiftKey){e.preventDefault(); sendMessage();}});
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index(): return HTML_CONTENT

@app.post("/chat")
async def chat_endpoint(message: str = Form(""), image: UploadFile = File(None)):
    try:
        # Agent xử lý việc gọi sang Vision và biên tập
        response_text = agent.chat_logic(message, image)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
