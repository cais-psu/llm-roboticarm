import openai
import base64
import os
import requests

# load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai.api_key}"
}

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the local images
image1_path = "C:\\Users\\jongh\\OneDrive - The Pennsylvania State University\\01_research\\02_LLM\\02. LLM-Assembly\\figures\\check3.png"
image2_path = "C:\\Users\\jongh\\OneDrive - The Pennsylvania State University\\01_research\\02_LLM\\02. LLM-Assembly\\figures\\check2.jpg"
encoded_image1 = encode_image_to_base64(image1_path)
encoded_image2 = encode_image_to_base64(image2_path)

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          #"text": "This is the last position before it picks up the part. Based on two images that are different angles of the same position, is the gripper located right above the obect to grasp it successfully do we need to adjust the position of the robotic arm? The widght of the part is 4.5cm and the gripper's width is 3cm. How much roughly do you think the gripper needs to move left looking from the side view (second picture) so that it is in the midpoint of the part that can grasp the part?"
          "text": "How far are they from each other roughly? can you calculate if i give you pix/mm?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image1}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image2}"
          }
        }        
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())