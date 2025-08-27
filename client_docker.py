"""
Client functions for calling the Multi-Whisper API
"""

import requests
import json
import soundfile as sf
import numpy as np
from typing import Optional


def speech_to_text_rest(
    audio_data: bytes,
    model: str = "openai_whisper",
    api_url: str = "http://localhost:8888",
) -> str:
    """
    Client function to call REST API for speech-to-text transcription

    Args:
        audio_data: Raw audio bytes (WAV format, 16kHz mono)
        model: Model to use ("openai_whisper" or "pho_whisper")
        api_url: Base URL of the transcription API

    Returns:
        Transcribed text
    """
    try:
        # Prepare the request
        files = {"file": ("test.wav", audio_data, "audio/wav")}
        params = {"model": model}

        # Make the API call
        response = requests.post(
            f"{api_url}/transcribe", files=files, params=params, timeout=30
        )

        if response.status_code == 200:
            print(f"Response: {response.json()}")
            result = response.json()
            return result.get("text", "")
        else:
            error_msg = (
                f"API call failed with status {response.status_code}: {response.text}"
            )
            raise Exception(error_msg)

    except Exception as e:
        print(f"Failed to transcribe speech to text: {str(e)}")
        raise e


def speech_to_text_websocket(
    audio_data: bytes, model: str = "whisper", api_url: str = "ws://localhost:8888"
) -> str:
    """
    Client function to call WebSocket API for speech-to-text transcription

    Args:
        audio_data: Raw audio bytes (WAV format, 16kHz mono)
        model: Model to use ("whisper" or "pho")
        api_url: Base WebSocket URL of the transcription API

    Returns:
        Transcribed text
    """
    try:
        import websockets
        import asyncio

        async def _transcribe():
            uri = f"{api_url}/ws/{model}"
            async with websockets.connect(
                uri,
                open_timeout=30,
                ping_interval=20,
                ping_timeout=20,
                max_size=20 * 1024 * 1024,
            ) as websocket:
                # Send audio data
                await websocket.send(audio_data)

                # Trigger transcription
                await websocket.send("END")

                # Wait for response
                response = await websocket.recv()
                print(f"Response: {response}")
                result = json.loads(response)

                if "text" in result:
                    return result["text"]
                elif "error" in result:
                    raise Exception(result["error"])
                else:
                    raise Exception("Unexpected response format")

        # Run the async function
        return asyncio.run(_transcribe())

    except Exception as e:
        print(f"Failed to transcribe speech to text via WebSocket: {str(e)}")
        raise e


def test_rest_api():
    """Test REST API endpoints"""
    print("=== Testing REST API ===")

    # Read audio file
    with open("test.wav", "rb") as f:
        audio_bytes = f.read()
    f.close()

    try:
        # Test OpenAI Whisper
        text_whisper = speech_to_text_rest(audio_bytes, "openai_whisper")
        print(f"OpenAI Whisper: {text_whisper}")

        # # Test PhoWhisper
        # text_pho = speech_to_text_rest(audio_bytes, "pho_whisper")
        # print(f"PhoWhisper: {text_pho}")

    except Exception as e:
        print(f"REST API Error: {e}")


def test_websocket_api():
    """Test WebSocket API endpoints"""
    print("=== Testing WebSocket API ===")

    # Create test audio
    with open("test.wav", "rb") as f:
        audio_bytes = f.read()

    try:
        # Test OpenAI Whisper WebSocket
        text_ws_whisper = speech_to_text_websocket(audio_bytes, "whisper")
        print(f"WebSocket OpenAI Whisper: {text_ws_whisper}")

        # # Test PhoWhisper WebSocket
        # text_ws_pho = speech_to_text_websocket(audio_bytes, "pho")
        # print(f"WebSocket PhoWhisper: {text_ws_pho}")

    except Exception as e:
        print(f"WebSocket API Error: {e}")


if __name__ == "__main__":
    # Run tests
    test_rest_api()
    print("--------------------------------")
    test_websocket_api()

    # Uncomment to test with real audio file
    # test_with_real_audio("path/to/your/audio.wav")
