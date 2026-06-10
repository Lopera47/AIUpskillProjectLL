# gemini_examples.py
import google.genai as genai
import os
from dotenv import load_dotenv

load_dotenv() 

model = 'gemini-2.5-flash'

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Example 1: Simple completion
response = client.models.generate_content(
    model=model,
    contents="Explain async/await in Python in one sentence"
)
print("Example 1:", response.text)

# Example 2: JSON output
prompt = """
Given this article title: "New AI Model Released"
Output JSON with these fields: relevant (boolean), reason (string)

{"relevant": true/false, "reason": "explanation"}
"""
response = client.models.generate_content(
    model=model,
    contents=prompt
)
print("\nExample 2:", response.text)

# Example 3: Few-shot learning
prompt = """
Classify articles as AI-related or not.

Examples:
Title: "GPT-4 Released" -> AI-related: Yes
Title: "Recipe for Pasta" -> AI-related: No
Title: "Machine Learning in Healthcare" -> AI-related: Yes

Now classify:
Title: "New JavaScript Framework"
"""
response = client.models.generate_content(
    model=model,
    contents=prompt
)
print("\nExample 3:", response.text)