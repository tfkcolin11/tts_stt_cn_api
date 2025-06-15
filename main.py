# main.py
import os
import shutil
import tempfile
import logging

# --- PyTorch Diagnostics ---
try:
    import torch
    logger_diag = logging.getLogger("pytorch_diag")
    logger_diag.info(f"Attempting to load PyTorch. Found version: {torch.__version__}")
    logger_diag.info(f"PyTorch installation location: {torch.__file__}")
    
    # Try a simple torch operation
    try:
        a = torch.zeros(1) # Attempt to use torch
        logger_diag.info(f"Successfully created a torch tensor: {a}. torch.zeros(1) works.")
    except Exception as e_tensor:
        logger_diag.error(f"Error creating a simple torch tensor: {e_tensor}", exc_info=True)

    # Test for the problematic attribute directly
    if hasattr(torch, 'get_default_device'):
        logger_diag.info("torch.get_default_device attribute IS PRESENT.")
    else:
        logger_diag.warning("torch.get_default_device attribute IS MISSING from the loaded torch module.")
except ImportError:
    logging.getLogger("pytorch_diag").error("PyTorch could not be imported at all!")
except Exception as e:
    logging.getLogger("pytorch_diag").error(f"Error during PyTorch diagnostics: {e}", exc_info=True)
# --- End PyTorch Diagnostics ---


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

# --- Configuration ---
DEVICE = "cpu" 
logger.info(f"Target device set to: {DEVICE}")

# STT Model (Whisper)
STT_MODEL_NAME = "openai/whisper-base"
stt_pipeline = None

# TTS Model (Coqui TTS for Chinese)
TTS_MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
tts_model = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global stt_pipeline, tts_model
    logger.info("Starting up and loading models...")

    # Re-import torch here to ensure it's the one used by transformers in this scope
    # This is mostly for safety, the top-level import should be sufficient.
    try:
        import torch
    except ImportError:
        logger.error("Failed to import PyTorch within startup_event. STT will likely fail.")
        # Optionally, prevent STT loading if torch isn't even importable here
        # return 

    # STT Model Loading
    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading STT model: {STT_MODEL_NAME}")

        # Determine device for Hugging Face pipeline
        hf_device_param = None
        if DEVICE == "cpu":
            hf_device_param = torch.device("cpu") # Use torch.device object
            logger.info(f"Explicitly setting Hugging Face STT pipeline device to: {hf_device_param}")
        else: # Assuming CUDA (or other devices in the future)
            if torch.cuda.is_available():
                hf_device_param = torch.device("cuda:0") # Use first GPU
                logger.info(f"Explicitly setting Hugging Face STT pipeline device to: {hf_device_param}")
            else:
                logger.warning(f"CUDA specified for STT but not available according to PyTorch. Falling back to CPU for Hugging Face pipeline.")
                hf_device_param = torch.device("cpu")
        
        stt_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL_NAME,
            device=hf_device_param # Pass the torch.device object
        )
        logger.info("STT model loaded successfully.")
    except AttributeError as ae:
        logger.error(f"AttributeError loading STT model ({STT_MODEL_NAME}): {ae}", exc_info=True)
        logger.error("This often indicates an issue with the PyTorch version or its installation, or the specific attribute not being found.")
        stt_pipeline = None
    except Exception as e:
        logger.error(f"Generic error loading STT model ({STT_MODEL_NAME}): {e}", exc_info=True)
        stt_pipeline = None

    # TTS Model Loading
    try:
        from TTS.api import TTS as CoquiTTS
        logger.info(f"Loading TTS model: {TTS_MODEL_NAME}")
        # Coqui TTS gpu parameter is boolean
        tts_gpu_param = False
        if DEVICE == "cuda":
            if torch.cuda.is_available(): # Check with PyTorch if CUDA is truly usable
                logger.info("CUDA is available. Setting Coqui TTS to use GPU.")
                tts_gpu_param = True
            else:
                logger.warning("CUDA specified for TTS but not available according to PyTorch. Coqui TTS will use CPU.")
        
        tts_model = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=True,
            gpu=tts_gpu_param # Pass boolean
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
        logger.warning(f"Received file with potentially invalid content type: {audio_file.content_type} for filename: {audio_file.filename}")

    tmp_audio_file_path = None
    try:
        suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio_file:
            shutil.copyfileobj(audio_file.file, tmp_audio_file)
            tmp_audio_file_path = tmp_audio_file.name
        logger.info(f"Temporary audio file for STT saved at: {tmp_audio_file_path}")

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
            except Exception as e_del_stt:
                logger.error(f"Error deleting temporary STT file {tmp_audio_file_path}: {e_del_stt}")
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output_file:
            output_wav_path = tmp_output_file.name
        
        logger.info(f"Generating TTS for text: '{request.text[:50]}...' to {output_wav_path}")
        
        tts_model.tts_to_file(
            text=request.text,
            file_path=output_wav_path
        )
        logger.info(f"TTS audio generated successfully: {output_wav_path}")

        return FileResponse(
            path=output_wav_path,
            media_type="audio/wav",
            filename="output.wav",
            background=BackgroundTask(lambda p: os.remove(p) if os.path.exists(p) else None, output_wav_path)
        )
    except Exception as e:
        logger.error(f"Error during TTS processing: {e}", exc_info=True)
        if output_wav_path and os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
                logger.info(f"Cleaned up temporary TTS file due to error: {output_wav_path}")
            except Exception as del_e:
                 logger.error(f"Error deleting temporary TTS file {output_wav_path} after error: {del_e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/")
async def root():
    torch_version_display = "N/A"
    torch_file_display = "N/A"
    torch_diag_present = "N/A"
    try:
        import torch
        torch_version_display = torch.__version__
        torch_file_display = torch.__file__
        torch_diag_present = "Present" if hasattr(torch, 'get_default_device') else "MISSING"
    except:
        pass

    return {
        "message": "Chinese STT/TTS API",
        "stt_status": "available" if stt_pipeline else "unavailable",
        "tts_status": "available" if tts_model else "unavailable",
        "stt_model_name": STT_MODEL_NAME if stt_pipeline else "N/A",
        "tts_model_name": TTS_MODEL_NAME if tts_model else "N/A",
        "pytorch_version": torch_version_display,
        "pytorch_location": torch_file_display,
        "pytorch_get_default_device_attr": torch_diag_present,
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)