import cv2
import base64
import os
import requests
import time
from openai import OpenAI
from collections import deque
from datetime import datetime

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

def send_frame_to_gpt(frame, previous_texts, client):
    context = ' '.join(previous_texts)
    #prompt_message = f"Context: {context}. Assess if the previous prediction matches the current situation. Current: explain the current situation in 10 words or less. Next: Predict the next situation in 10 words or less."
    prompt_message = f"Context: {context}. Assess if the previous prediction matches the current situation."
    PROMPT_MESSAGES = {
        "role": "user",
        "content": [
            prompt_message,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        ],
    }

    params = {
        "model": "gpt-4-vision-preview",
        "messages": [PROMPT_MESSAGES],
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def main():
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    video = cv2.VideoCapture(0)
    previous_texts = deque(maxlen=5)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        base64_image = encode_image_to_base64(frame)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        generated_text = send_frame_to_gpt(base64_image, previous_texts, client)
        print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")
        previous_texts.append(f"[{timestamp}] {generated_text}")
        time.sleep(5)
    video.release()

if __name__ == "__main__":
    main()