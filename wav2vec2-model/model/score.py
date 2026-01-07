import json
import os
import io
import torch
import numpy as np
import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Global variables (loaded once)
processor = None
model = None
device = None


def init():
    """
    Called once when the container is started.
    Load the wav2vec2 model and processor here.
    """
    global processor, model, device

    model_name = "facebook/wav2vec2-base-960h"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    model.to(device)
    model.eval()

    print("wav2vec2 model loaded successfully")


def run(raw_data):
    """
    Called every time an inference request is made.
    Expects raw audio bytes.
    """
    try:
        # raw_data is bytes when sent as application/octet-stream
        audio_bytes = io.BytesIO(raw_data)

        # Read audio
        speech, sample_rate = sf.read(audio_bytes)

        # Convert stereo to mono if needed
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)

        # Preprocess
        inputs = processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return {
            "transcription": transcription
        }

    except Exception as e:
        return {
            "error": str(e)
        }
