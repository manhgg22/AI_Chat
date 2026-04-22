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
        Extract structured data from image.
        """
        print(f"[VISION PROCESSOR] Extracting structured data for: {image_path}")
        
        base64_image = self._encode_image(image_path)
        
        # SYSTEM PROMPT FOR STRUCTURED EXTRACTION
        structured_prompt = f"""
        Analyze the image and provide a structured JSON response. 
        Focus on:
        1. Context Category: (Document, Scene, Product, Text, Face, etc.)
        2. Detailed Items: List of objects/elements detected.
        3. OCR: Exact text found in the image.
        4. Technical attributes: Colors, lighting, resolution feel.
        5. Confidence Score: (0-1.0) based on how clear the image is.
        
        User's specific interest: {user_prompt}
        
        Return EXCLUSIVELY a JSON object with this keys: 
        {{
            "category": "string",
            "detected_elements": ["list"],
            "ocr_content": "string",
            "visual_description": "string",
            "confidence": float,
            "warning": "string if confidence < 0.7 else ''"
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
