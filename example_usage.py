from vision_tool import GroqVisionTool
import os

def main():
    # 1. Initialize the tool
    # Make sure you have GROQ_API_KEY set in your environment or .env file
    try:
        vision = GroqVisionTool()
    except ValueError as e:
        print(f"Setup Error: {e}")
        return

    # 2. Define your query and image
    prompt = "Describe the contents of this image in detail."
    
    # URL Example
    image_url = "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
    
    print(f"--- Analyzing Image (URL) ---")
    print(f"Prompt: {prompt}")
    print(f"Image: {image_url}")
    
    # In a real scenario, this would call the API. 
    # For now, we'll just print instructions since the key might not be set.
    if os.environ.get("GROQ_API_KEY"):
        result = vision.analyze_image(prompt, image_url)
        print(f"\nResult:\n{result}")
    else:
        print("\n[ACTION REQUIRED] Please set GROQ_API_KEY in your .env file to run this example.")

if __name__ == "__main__":
    main()
