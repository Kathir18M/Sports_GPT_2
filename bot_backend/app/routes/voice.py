import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ai.voice import speech_to_text, text_to_speech

router = APIRouter()

TEMP_DIR = "recordings"

@router.post("/voice")
async def process_voice(file: UploadFile = File(...)):
    """
    Receives an audio recording file from the frontend,
    transcribes it, generates an AI answer, generates TTS audio,
    and returns transcript, text response, and audio URL.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file uploaded")

    os.makedirs(TEMP_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".webm"
    temp_filename = f"upload_{uuid.uuid4().hex}{ext}"
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        stt_result = await speech_to_text(temp_filepath)
        transcript = stt_result["text"] if isinstance(stt_result, dict) else stt_result[0]

        from app.ai.llm import generate_response
        llm_res = generate_response(transcript, "fast")
        answer = llm_res.get("response", "No response") if isinstance(llm_res, dict) else str(llm_res)

        tts_result = await text_to_speech(answer)
        output_audio_path = tts_result["audio_path"] if isinstance(tts_result, dict) else tts_result

        filename = os.path.basename(output_audio_path)
        audio_url = f"/outputs/{filename}"

        return {
            "transcript": transcript,
            "response": answer,
            "audio_url": audio_url,
            "provider": "Google AI",
            "model": "Gemini 2.5 Flash"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
