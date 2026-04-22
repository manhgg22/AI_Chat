import os
import json
from typing import Optional, Dict, List
from groq import Groq
from dotenv import load_dotenv
from vision_processor import VisionProcessor

load_dotenv()

class ProfessionalAgent:
    def __init__(self, model: str = "qwen/qwen3-32b", **kwargs):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.vision_processor = VisionProcessor()

    def generate_final_response(self, user_query: str, image_paths: Optional[List[str]] = None) -> str:
        """
        Orchestrate the vision-only professional pipeline.
        """
        if not image_paths:
            # Simple text response for general queries
            return self._simple_chat(user_query)

        # 1. Vision Layer: Structured Extraction
        print(f"[AGENT] Activating Vision Layer for {len(image_paths)} images...", flush=True)
        aggregated_vision_data = []
        total_confidence = 0
        
        for idx, path in enumerate(image_paths):
            print(f"[AGENT] Processing Image {idx+1}/{len(image_paths)}...", flush=True)
            vision_data = self.vision_processor.process_image(path, user_query)
            aggregated_vision_data.append({
                "image_index": idx + 1,
                "data": vision_data
            })
            total_confidence += vision_data.get('confidence', 0)
        
        avg_confidence = total_confidence / len(image_paths)
        
        # 2. Intelligence Layer: Post-processing and Formatting
        print("[AGENT] Activating Intelligence Layer (Post-processing)...", flush=True)
        
        system_prompt = f"""
        Bạn là Thẩm phán Phân tích Hình ảnh Tối cao.
        
        BỐI CẢNH: Bạn nhận được dữ liệu từ Vision Model cho {len(image_paths)} hình ảnh. 
        CÂU HỎI: "{user_query}"
        
        CHỈ THỊ BẮT BUỘC (QUYẾT ĐỊNH VẬN MỆNH):
        1. TUYỆT ĐỐI KHÔNG được trả lời mập mờ, nước đôi hoặc dùng các cụm từ như "không thể xác định", "có vẻ như", "dường như", "có khả năng".
        2. Bạn PHẢI đưa ra phán quyết CUỐI CÙNG: "ĐÚNG/CÓ" hoặc "SAI/KHÔNG". 
        3. Dựa trên dữ liệu JSON, hãy tìm ra những điểm tương đồng hoặc khác biệt dù là nhỏ nhất để đưa ra lựa chọn có xác suất cao nhất. 
        4. Sau khi chốt kết luận (ví dụ: "KẾT LUẬN: ĐÂY LÀ CÙNG MỘT NGƯỜI"), hãy liệt kê các bằng chứng kỹ thuật từ dữ liệu (ví dụ: cấu trúc khuôn mặt, OCR, đặc điểm nhận dạng) để bảo vệ phán quyết của mình.
        5. Nếu độ tin cậy của dữ liệu thấp, bạn vẫn phải chọn một bên có lý nhất và khẳng định nó, kèm theo một lời cảnh báo kỹ thuật rất ngắn ở cuối.
        
        Hãy nhớ: Người dùng cần một câu trả lời dứt khoát để hành động. Đừng làm họ thất vọng bằng sự do dự!
        """
        
        context_payload = f"""
        DỮ LIỆU TỔNG HỢP TỪ VISION MODEL ({len(image_paths)} ảnh):
        {json.dumps(aggregated_vision_data, indent=2, ensure_ascii=False)}
        """
        print(f"[AGENT] Context Payload for Intelligence Layer:\n{context_payload}", flush=True)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_payload}
                ],
                temperature=0.3,
                max_completion_tokens=2048
            )
            
            # Clean up the <think> block if present
            import re
            response = completion.choices[0].message.content
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            # Append Confidence Badge
            confidence_pct = avg_confidence * 100
            badge = f"\n\n---\n**Độ tin cậy hệ thống (trung bình):** {confidence_pct:.1f}%"
            if avg_confidence < 0.7:
                badge += f"\n⚠ **Cảnh báo:** Kết quả tổng hợp có thể chưa hoàn toàn chính xác do chất lượng ảnh đầu vào."
            
            return response + badge

        except Exception as e:
            return f"Lỗi xử lý hậu kỳ: {str(e)}"

    def _simple_chat(self, query: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý ảo tiếng Việt chuyên nghiệp."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            import re
            response = completion.choices[0].message.content
            return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        except Exception as e:
            return f"Lỗi kết nối API: {str(e)}"

    def chat(self, user_query: str, image_paths: Optional[List[str]] = None) -> str:
        """Alias for generate_final_response to maintain compatibility."""
        return self.generate_final_response(user_query, image_paths)

# Alias to maintain compatibility with existing chat_server.py
ChatAgent = ProfessionalAgent
