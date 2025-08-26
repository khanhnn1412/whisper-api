from fastapi import FastAPI, UploadFile, File, WebSocket, Query
from fastapi.websockets import WebSocketDisconnect
from transformers import pipeline
import whisper
import soundfile as sf
import numpy as np
import io
from enum import Enum
from typing import Optional

import torch

app = FastAPI(title="Multi-Whisper Service")


class WhisperModel(str, Enum):
    PHO_WHISPER = "pho_whisper"
    OPENAI_WHISPER = "openai_whisper"


# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_device = 0 if device == "cuda" else -1

# PhoWhisper model
pho_whisper_model = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-large",
    device=hf_device,
    torch_dtype=torch.float16 if device == "cuda" else None,
)

# OpenAI Whisper model
openai_whisper_model = whisper.load_model("large-v3-turbo", device=device)


def transcribe_audio(
    audio_data: np.ndarray, sample_rate: int, model_type: WhisperModel
) -> str:
    """Transcribe audio using the specified model"""
    if model_type == WhisperModel.PHO_WHISPER:
        result = pho_whisper_model(audio_data, sampling_rate=sample_rate)
        return result["text"]
    elif model_type == WhisperModel.OPENAI_WHISPER:
        result = openai_whisper_model.transcribe(audio_data)
        return result["text"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: WhisperModel = Query(
        default=WhisperModel.OPENAI_WHISPER, description="Choose transcription model"
    ),
):
    """Transcribe audio file using specified model"""
    # read audio
    wav, sr = sf.read(file.file)

    # Whisper expects 16kHz audio, so we need to resample if necessary
    if sr != 16000:
        return {"error": "Audio must be 16kHz mono"}

    if wav.ndim > 1:  # stereo -> mono
        wav = np.mean(wav, axis=1)

    # Transcribe using selected model
    try:
        text = transcribe_audio(wav, sr, model)
        return {
            "text": text,
            "model": model.value,
            "sample_rate": sr,
            "duration": len(wav) / sr,
        }
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


async def handle_websocket_transcription(
    websocket: WebSocket, model_type: WhisperModel
):
    """Common WebSocket handler for transcription"""
    await websocket.accept()
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            data_bytes = message.get("bytes")
            data_text = message.get("text")

            if data_bytes is not None:
                # Accumulate audio bytes
                audio_buffer.extend(data_bytes)
                await websocket.send_json(
                    {
                        "status": "buffered",
                        "buffer_bytes": len(audio_buffer),
                        "model": model_type.value,
                    }
                )
                continue

            if data_text is not None:
                text_cmd = data_text.strip().lower()

                if text_cmd == "reset":
                    audio_buffer.clear()
                    await websocket.send_json(
                        {"status": "reset", "model": model_type.value}
                    )
                    continue

                if text_cmd == "end":
                    if not audio_buffer:
                        await websocket.send_json({"error": "no audio buffered"})
                        continue

                    wav: np.ndarray
                    sr: int = 16000

                    # Try to decode as WAV first
                    try:
                        with io.BytesIO(bytes(audio_buffer)) as wav_io:
                            wav, sr = sf.read(wav_io)
                    except Exception:
                        # Fallback: assume raw PCM int16 LE mono @16kHz
                        wav_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        wav = wav_np.astype(np.float32) / 32768.0
                        sr = 16000

                    # Validate and preprocess
                    if sr != 16000:
                        await websocket.send_json(
                            {"error": "audio must be 16kHz; got %d" % sr}
                        )
                        audio_buffer.clear()
                        continue

                    if wav.ndim > 1:
                        wav = np.mean(wav, axis=1)

                    # Run transcription using selected model
                    try:
                        text = transcribe_audio(wav, sr, model_type)
                        await websocket.send_json(
                            {
                                "text": text,
                                "model": model_type.value,
                                "sample_rate": sr,
                                "duration": len(wav) / sr,
                            }
                        )
                    except Exception as e:
                        await websocket.send_json(
                            {
                                "error": f"Transcription failed: {str(e)}",
                                "model": model_type.value,
                            }
                        )

                    # Clear buffer for next session
                    audio_buffer.clear()
                    continue

                # Unknown text command
                await websocket.send_json(
                    {
                        "status": "ignored",
                        "message": data_text,
                        "model": model_type.value,
                    }
                )

    except WebSocketDisconnect:
        # Client disconnected; just exit gracefully
        return


@app.websocket("/ws/whisper")
async def ws_whisper(websocket: WebSocket) -> None:
    """WebSocket endpoint for OpenAI Whisper transcription.

    Protocol:
    - Client sends binary frames containing audio bytes. You can stream either:
      1) a complete WAV file split across frames, or
      2) raw PCM 16-bit little-endian mono at 16 kHz.
    - Send a text frame "END" to trigger transcription of buffered audio.
    - Send a text frame "RESET" to clear the current buffer without transcribing.
    - Server responds with JSON messages for status and final transcription.
    """
    await handle_websocket_transcription(websocket, WhisperModel.OPENAI_WHISPER)


@app.websocket("/ws/pho")
async def ws_pho(websocket: WebSocket) -> None:
    """WebSocket endpoint for PhoWhisper transcription.

    Protocol:
    - Client sends binary frames containing audio bytes. You can stream either:
      1) a complete WAV file split across frames, or
      2) raw PCM 16-bit little-endian mono at 16 kHz.
    - Send a text frame "END" to trigger transcription of buffered audio.
    - Send a text frame "RESET" to clear the current buffer without transcribing.
    - Server responds with JSON messages for status and final transcription.
    """
    await handle_websocket_transcription(websocket, WhisperModel.PHO_WHISPER)


@app.get("/models")
async def get_available_models():
    """Get list of available transcription models"""
    return {
        "models": [
            {
                "id": model.value,
                "name": (
                    "PhoWhisper Large"
                    if model == WhisperModel.PHO_WHISPER
                    else "OpenAI Whisper Base"
                ),
                "description": (
                    "Vietnamese-optimized Whisper model"
                    if model == WhisperModel.PHO_WHISPER
                    else "Multilingual Whisper model"
                ),
                "websocket_endpoint": f"/ws/{model.value.split('_')[0]}",
            }
            for model in WhisperModel
        ]
    }
