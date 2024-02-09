# import needed functions from pvrecorder
from pvrecorder import PvRecorder
import wave, struct

# identify and print available audio devices
for index, device in enumerate(PvRecorder.get_available_devices()):
    print(f"[{index}] {device}")

# choose audio device to record with, use -1 for index to use system default
recorder = PvRecorder(device_index=0, frame_length=512)
audio = []

# names audio file
path = 'audio_recording.wav'

# starts recording when the cell is run, stops recording when the cell is stopped
try:
    recorder.start()  
    while True:
        frame = recorder.read()
        audio.extend(frame)
except KeyboardInterrupt:
    recorder.stop()
    with wave.open(path, 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
finally:
    recorder.delete()

# Opened audio file that I recorded on my computer and uploaded to Google Colab
audio_file= open("audio_recording.wav", "rb")

# Opens and reads audio file to traslate it into text
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  response_format="text"
)

# Prints result
print(transcript)

# imports functions from OpenAI
from openai import OpenAI

# Sets API key; each use costs credits
client = OpenAI(api_key = "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")

# specifies model to be used, sets repsonse format to JSON for easier comprehension, 
#   and bases answer on transcript from recorded audio
response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": transcript}
  ]
)
print(response.choices[0].message.content)

# Converts chat gpt response to 
soundoutput = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input= response.choices[0].message.content,
)

soundoutput.stream_to_file("rtaconvert.mp3")