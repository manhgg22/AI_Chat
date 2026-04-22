import base64
import os
from typing import Optional, Union
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GroqVisionTool:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq Vision Tool.
        :param api_key: Groq API Key. If None, it will look for GROQ_API_KEY in environment variables.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key not found. Please provide it or set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _encode_image(self, image_path: str) -> str:
        """
        Encodes a local image to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, prompt: str, image_source: str, lang: str = "Vietnamese") -> str:
        """
        Analyze an image using Groq's Llama 4 Scout Vision model.
        
        :param prompt: The question or instruction for the model.
        :param image_source: Can be a URL (starting with http/https) or a local file path.
        :param lang: The language for the response (default is Vietnamese).
        :return: The model's response text.
        """
        # Append language instruction to the prompt
        full_prompt = f"{prompt}\n\nPlease provide your response in {lang}."
        
        image_url = ""
        
        if image_source.startswith("http://") or image_source.startswith("https://"):
            # It's a URL
            image_url = image_source
        else:
            # It's a local path
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Local image file not found: {image_source}")
            
            # Determine mime type (basic check)
            ext = os.path.splitext(image_source)[1].lower()
            mime_type = "image/jpeg"
            if ext == ".png":
                mime_type = "image/png"
            elif ext == ".webp":
                mime_type = "image/webp"
            
            base64_image = self._encode_image(image_source)
            image_url = f"data:{mime_type};base64,{base64_image}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    } 
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error during vision analysis: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Test with a URL (using the example from documentation)
    # tool = GroqVisionTool()
    # response = tool.analyze_image("What's in this image?", "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg")
    # print(response)
    print("GroqVisionTool is ready. Configure GROQ_API_KEY in your .env file.")
