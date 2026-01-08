# Azure ML HuBERT Speech-to-Text

This project deploys a HuBERT (facebook-hubert-base-ls960) model for speech-to-text inference using Azure Machine Learning.

## Setup

1. **Configure Azure ML credentials:**
   ```bash
   cp config.template.json config.json
   ```
   Then edit `config.json` with your Azure subscription details.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to Azure:**
   ```bash
   az login
   ```

## Deployment

Deploy the model to Azure ML managed endpoint:

```bash
cd wav2vec2-model/model
az ml online-deployment create \
  --file deployment.yml \
  --workspace-name wav2vec2-aml \
  --resource-group wav2vec2-rg
```

Or use the Jupyter notebook `HF.ipynb` which automates the entire process including model download from HuggingFace registry.

## Project Structure

- `HF.ipynb` - Notebook for model download and deployment
- `wav2vec2-model/` - Model artifacts and deployment configuration (legacy name, now contains HuBERT)
  - `model/score.py` - Scoring script for inference using HuBERT
  - `model/deployment.yml` - Azure ML deployment configuration
- `requirements.txt` - Python dependencies

## Notes

- Model files (`.safetensors`) are downloaded from HuggingFace Hub (facebook/hubert-base-ls960)
- The deployment uses `Standard_F2s_v2` instance. For production, consider `Standard_DS3_v2` or larger.
- HuBERT model is similar to Wav2Vec2 but pre-trained differently for improved speech recognition.
# azure-hubert
