# AI Chat

Chat agent chạy trên FastAPI, dùng Groq làm LLM, kèm công cụ xử lý hình ảnh.

## Thành phần

| File | Vai trò |
|---|---|
| `app.py` | FastAPI app chính — endpoint chat và upload ảnh |
| `chat_agent.py` | Logic agent, quản lý hội thoại và gọi tool |
| `chat_server.py` | Server chat tách riêng |
| `vision_processor.py` / `vision_service.py` / `vision_tool.py` | Xử lý ảnh và expose thành tool cho agent |
| `chat_ui.html` | Giao diện chat |

## Chạy

```bash
pip install -r requirements.txt
cp .env.example .env   # điền GROQ_API_KEY
python app.py
```

Mặc định phục vụ qua uvicorn. Mở `chat_ui.html` hoặc truy cập root để dùng giao diện.
