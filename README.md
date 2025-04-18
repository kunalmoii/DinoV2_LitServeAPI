# DinoV2_LitServeAPI
DinoV2 LitServe API for inference on cloud

To Pull the Data run:
```bash
export GOOGLE_APPLICATION_CREDENTIALS='credentials/gcp_key.json'
dvc pull required_name.dvc
```


docker build -t dinov2-litserve .
docker run --gpus all -p 8000:8000 dinov2-litserve
