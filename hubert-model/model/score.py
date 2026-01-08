import os
import logging
import json
import torch
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor

def init():
    """
    This function is called when the container is initialized/started, usually after driver/Docker startup.
    We load the model here so that it is only loaded once.
    """
    global model, feature_extractor
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.getenv("AZUREML_MODEL_DIR")
    
    # Check if model files are in a nested 'model' subdirectory
    nested_model_path = os.path.join(model_path, "model")
    if os.path.exists(nested_model_path) and os.path.isdir(nested_model_path):
        model_path = nested_model_path
    
    logging.info(f"Loading model from: {model_path}")
    
    try:
        # Load the feature extractor and model from the local model directory
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = HubertModel.from_pretrained(model_path)
        
        logging.info("HuBERT model and feature extractor loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise e

def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    """
    try:
        logging.info("Received request")
        
        # Parse input (JSON with 'audio' key containing base64 encoded audio)
        data = json.loads(raw_data)
        
        if 'audio' not in data:
            return {"error": "No 'audio' field in request"}
        
        # Decode base64 audio
        import base64
        import io
        import soundfile as sf
        
        audio_base64 = data['audio']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Read audio using soundfile
        audio_buffer = io.BytesIO(audio_bytes)
        speech_array, sample_rate = sf.read(audio_buffer)
        
        # Resample to 16000 Hz if needed (HuBERT expects 16kHz)
        if sample_rate != 16000:
            import librosa
            speech_array = librosa.resample(speech_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process audio with HuBERT
        inputs = feature_extractor(speech_array, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(inputs.input_values)
        
        # Extract hidden states (features)
        hidden_states = outputs.last_hidden_state
        
        # Return the embeddings
        embeddings = hidden_states.mean(dim=1).squeeze().tolist()
        
        logging.info(f"Generated embeddings of length: {len(embeddings)}")
        
        return {"status": "success", "embeddings": embeddings}
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}
