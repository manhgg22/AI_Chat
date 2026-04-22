import base64
import os
import json
from typing import Optional, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class VisionProcessor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_image(self, image_path: str, user_prompt: str) -> Dict:
        """
        Extract structured data from image using the user-specified scout model.
        """
        print(f"[VISION PROCESSOR] Extracting structured data for: {image_path}", flush=True)
        
        base64_image = self._encode_image(image_path)
        
        # SYSTEM PROMPT FOR DETAILED SCOUT ANALYSIS
        structured_prompt = f"""
        Analyze this image in detail and provide a structured JSON response.
        
        User's specific interest: {user_prompt}
        
        Return EXCLUSIVELY a JSON object with these keys: 
        {{
            "category": "string",
            "detected_elements": ["list of objects/features"],
            "ocr_content": "string",
            "visual_description": "detailed description",
            "confidence": float (0.0-1.0),
            "warning": "string"
        }}
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": structured_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                temperature=0.1, # Keep it deterministic
                response_format={"type": "json_object"}
            )
            raw_result = completion.choices[0].message.content
            print(f"[VISION PROCESSOR] Raw result: {raw_result}", flush=True)
            return json.loads(raw_result)
        except Exception as e:
            print(f"[VISION PROCESSOR] Error: {e}")
            return {
                "category": "unknown",
                "detected_elements": [],
                "ocr_content": "",
                "visual_description": f"Error during processing: {str(e)}",
                "confidence": 0.0,
                "warning": "Critical error in vision pipeline."
            }
