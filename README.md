# Azure ML Wav2Vec2 Speech-to-Text

This project deploys a Wav2Vec2 model for speech-to-text inference using Azure Machine Learning.

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
  --workspace-name <your-workspace> \
  --resource-group <your-resource-group>
```

## Project Structure

- `HF.ipynb` - Notebook for model training/experimentation
- `wav2vec2-model/` - Model artifacts and deployment configuration
  - `model/score.py` - Scoring script for inference
  - `model/deployment.yml` - Azure ML deployment configuration
- `requirements.txt` - Python dependencies

## Notes

- Model files (`.safetensors`) are not included in the repository due to size. Register your model in Azure ML or download from Hugging Face.
- The deployment uses `Standard_DS1_v2` instance. For production, consider `Standard_DS3_v2` or larger.
