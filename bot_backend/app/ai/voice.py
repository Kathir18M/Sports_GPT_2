import os
import time
import edge_tts

from litellm import transcription
from app.core.config import settings


# ==========================================
# Output Folder
# ==========================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Speech to Text (Groq Whisper)
# ==========================================

async def speech_to_text(audio_path: str):

    print("\n Starting Voice-to-Text...\n")

    start_time = time.perf_counter()

    with open(audio_path, "rb") as audio_file:

        response = transcription(
            model="groq/whisper-large-v3-turbo",
            file=audio_file,
            api_key=settings.GROQ_API_KEY,
        )

    end_time = time.perf_counter()

    latency_ms = round((end_time - start_time) * 1000, 2)

    print("=" * 50)
    print("VOICE TO TEXT")
    print("=" * 50)
    print(f"Provider   : Groq")
    print(f"Model      : whisper-large-v3-turbo")
    print(f"Latency    : {latency_ms} ms")
    print(f"Transcript : {response.text}")
    print("=" * 50)

    return {
        "text": response.text,
        "latency_ms": latency_ms,
        "provider": "Groq",
        "model": "whisper-large-v3-turbo"
    }



async def text_to_speech(text: str):

    print("\n🔊 Starting Text-to-Speech...\n")

    start_time = time.perf_counter()

    filename = None

    for i in range(1, 100000):

        path = os.path.join(
            OUTPUT_DIR,
            f"response_{i}.mp3"
        )

        if not os.path.exists(path):
            filename = path
            break

    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-AriaNeural"
    )

    await communicate.save(filename)

    end_time = time.perf_counter()

    latency_ms = round((end_time - start_time) * 1000, 2)

    print("=" * 50)
    print("TEXT TO SPEECH")
    print("=" * 50)
    print(f"Latency : {latency_ms} ms")
    print(f"Saved   : {filename}")
    print("=" * 50)

    return {
        "audio_path": filename,
        "latency_ms": latency_ms
    }