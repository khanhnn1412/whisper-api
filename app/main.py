from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.websockets import WebSocketDisconnect
from transformers import pipeline
import soundfile as sf
import numpy as np
import io

import torch

app = FastAPI(title="PhoWhisper Service")

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_device = 0 if device == "cuda" else -1
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-large",
    device=hf_device,
    torch_dtype=torch.float16 if device == "cuda" else None,
)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # read audio
    wav, sr = sf.read(file.file)
    if sr != 16000:
        return {"error": "Audio must be 16kHz mono"}

    if wav.ndim > 1:  # stereo -> channel 1
        wav = np.mean(wav, axis=1)

    result = transcriber(wav, sampling_rate=sr)
    return {"text": result["text"]}


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """WebSocket endpoint to receive streaming audio and transcribe.

    Protocol:
    - Client sends binary frames containing audio bytes. You can stream either:
      1) a complete WAV file split across frames, or
      2) raw PCM 16-bit little-endian mono at 16 kHz.
    - Send a text frame "END" to trigger transcription of buffered audio.
    - Send a text frame "RESET" to clear the current buffer without transcribing.
    - Server responds with JSON messages for status and final transcription.
    """

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
                    {"status": "buffered", "buffer_bytes": len(audio_buffer)}
                )
                continue

            if data_text is not None:
                text_cmd = data_text.strip().lower()

                if text_cmd == "reset":
                    audio_buffer.clear()
                    await websocket.send_json({"status": "reset"})
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

                    # Run transcription
                    result = transcriber(wav, sampling_rate=sr)
                    await websocket.send_json({"text": result.get("text", "")})

                    # Clear buffer for next session
                    audio_buffer.clear()
                    continue

                # Unknown text command
                await websocket.send_json({"status": "ignored", "message": data_text})

    except WebSocketDisconnect:
        # Client disconnected; just exit gracefully
        return
