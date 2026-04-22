import os
import json
from typing import Optional, Dict
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

    def generate_final_response(self, user_query: str, image_path: Optional[str] = None) -> str:
        """
        Orchestrate the vision-only professional pipeline.
        """
        if not image_path:
            # Simple text response for general queries
            return self._simple_chat(user_query)

        # 1. Vision Layer: Structured Extraction
        print("[AGENT] Activating Vision Layer...")
        vision_data = self.vision_processor.process_image(image_path, user_query)
        
        # 2. Intelligence Layer: Post-processing and Formatting
        print("[AGENT] Activating Intelligence Layer (Post-processing)...")
        
        system_prompt = f"""
        Bạn là Chuyên gia Phân tích Hình ảnh. Nhiệm vụ của bạn là nhận dữ liệu trích xuất từ Vision Model (JSON) và trình bày lại cho người dùng theo yêu cầu của họ.
        
        QUY TẮC:
        1. Trình bày bằng tiếng Việt chuyên nghiệp, rõ ràng.
        2. Nếu độ tin cậy (confidence) thấp (<0.7), PHẢI đưa ra lời cảnh báo (Warning).
        3. Phân biệt rõ các phần: Quan sát chính, Nội dung văn bản (nếu có), và Kết luận.
        4. Trả lời trực tiếp vào câu hỏi người dùng: "{user_query}"
        """
        
        context_payload = f"""
        DỮ LIỆU TỪ VISION MODEL:
        {json.dumps(vision_data, indent=2, ensure_ascii=False)}
        """

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
            confidence_val = vision_data.get('confidence', 0) * 100
            badge = f"\n\n---\n**Độ tin cậy của hệ thống:** {confidence_val}%"
            if confidence_val < 70:
                badge += f"\n⚠ **Cảnh báo:** {vision_data.get('warning', 'Kết quả có thể chưa hoàn toàn chính xác.')}"
            
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

# Alias to maintain compatibility with existing chat_server.py
ChatAgent = ProfessionalAgent
