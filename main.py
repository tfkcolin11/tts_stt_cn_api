# main.py
import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask # Correct import for BackgroundTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEVICE = "cpu" # Forcing CPU for wider compatibility. Change to "cuda" for GPU.
logger.info(f"Using device: {DEVICE}")

# STT Model (Whisper)
# You can change to "openai/whisper-small", "openai/whisper-medium" for better accuracy
# but higher resource usage.
STT_MODEL_NAME = "openai/whisper-base"
stt_pipeline = None

# TTS Model (Coqui TTS for Chinese)
# This model is specifically for Chinese and doesn't require a speaker_wav.
TTS_MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
tts_model = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global stt_pipeline, tts_model
    logger.info("Starting up and loading models...")

    try:
        from transformers import pipeline as hf_pipeline # Local import to ensure it's attempted at startup
        logger.info(f"Loading STT model: {STT_MODEL_NAME}")
        stt_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL_NAME,
            device=-1 if DEVICE == "cpu" else 0 # Transformers pipeline uses -1 for CPU, 0 for first GPU
        )
        logger.info("STT model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading STT model ({STT_MODEL_NAME}): {e}", exc_info=True)
        stt_pipeline = None

    try:
        from TTS.api import TTS as CoquiTTS # Local import
        logger.info(f"Loading TTS model: {TTS_MODEL_NAME}")
        tts_model = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=True,
            gpu=(DEVICE == "cuda") # Coqui TTS uses gpu=True/False
        )
        logger.info("TTS model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading TTS model ({TTS_MODEL_NAME}): {e}", exc_info=True)
        tts_model = None

    if stt_pipeline is None:
        logger.warning("STT model failed to load. STT endpoint will not be available.")
    if tts_model is None:
        logger.warning("TTS model failed to load. TTS endpoint will not be available.")

class TTSRequest(BaseModel):
    text: str

@app.post("/api/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    if stt_pipeline is None:
        raise HTTPException(status_code=503, detail="STT model is not available. Check server logs.")

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        # Fallback check if content_type is None or empty
        # You might want to be more strict or use a library to determine file type from content
        logger.warning(f"Received file with potentially invalid content type: {audio_file.content_type} for filename: {audio_file.filename}")
        # Allowing processing to continue but logging it. For production, more robust validation is needed.
        # raise HTTPException(status_code=400, detail="Invalid or missing audio file content type. Please upload an audio file (e.g., WAV, MP3).")


    # Create a temporary file to store the uploaded audio
    tmp_audio_file_path = None
    try:
        # Ensure a common extension like .wav or .mp3 for processing, ffmpeg (used by whisper) should handle various inputs
        suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        if not suffix: # Handle cases where filename might not have an extension
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio_file:
            shutil.copyfileobj(audio_file.file, tmp_audio_file)
            tmp_audio_file_path = tmp_audio_file.name
        logger.info(f"Temporary audio file for STT saved at: {tmp_audio_file_path}")

        # Perform STT
        # Whisper pipeline can handle file paths and various audio formats (if ffmpeg is installed).
        # It will also handle resampling to the required 16kHz.
        transcription_result = stt_pipeline(tmp_audio_file_path)
        transcribed_text = transcription_result.get("text", "").strip()
        logger.info(f"Transcription successful for {tmp_audio_file_path}: '{transcribed_text[:50]}...'")

        return {"text": transcribed_text}
    except Exception as e:
        logger.error(f"Error during STT processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.remove(tmp_audio_file_path)
                logger.info(f"Temporary STT audio file deleted: {tmp_audio_file_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary STT file {tmp_audio_file_path}: {e}")
        if audio_file:
            await audio_file.close()


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is not available. Check server logs.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    output_wav_path = None
    try:
        # Create a temporary file for the TTS output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output_file:
            output_wav_path = tmp_output_file.name
        
        logger.info(f"Generating TTS for text: '{request.text[:50]}...' to {output_wav_path}")
        
        # Generate speech
        tts_model.tts_to_file(
            text=request.text,
            file_path=output_wav_path
        )
        logger.info(f"TTS audio generated successfully: {output_wav_path}")

        # Return the generated WAV file
        # The file will be deleted after being sent by FileResponse thanks to the background task
        return FileResponse(
            path=output_wav_path,
            media_type="audio/wav",
            filename="output.wav", # Filename for the client
            background=BackgroundTask(lambda p: os.remove(p) if os.path.exists(p) else None, output_wav_path)
        )
    except Exception as e:
        logger.error(f"Error during TTS processing: {e}", exc_info=True)
        if output_wav_path and os.path.exists(output_wav_path): # Clean up if error before FileResponse
            try:
                os.remove(output_wav_path)
                logger.info(f"Cleaned up temporary TTS file due to error: {output_wav_path}")
            except Exception as del_e:
                 logger.error(f"Error deleting temporary TTS file {output_wav_path} after error: {del_e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Chinese STT/TTS API",
        "stt_status": "available" if stt_pipeline else "unavailable",
        "tts_status": "available" if tts_model else "unavailable",
        "stt_model_name": STT_MODEL_NAME if stt_pipeline else "N/A",
        "tts_model_name": TTS_MODEL_NAME if tts_model else "N/A",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # This part is for local development without Docker, if needed.
    # The Docker CMD will run uvicorn directly.
    # Manually trigger startup event for local dev if not using uvicorn programmatically with app object
    # For uvicorn CLI, startup events are handled automatically.
    uvicorn.run(app, host="0.0.0.0", port=8000)