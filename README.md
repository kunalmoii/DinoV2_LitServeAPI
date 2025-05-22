# DinoV2_LitServeAPI
DinoV2 LitServe API for inference on cloud

To Pull the Data run:
```bash
export GOOGLE_APPLICATION_CREDENTIALS='credentials/gcp_key.json'
dvc pull checkpoints.dvc
```

To Build Docker:
```bash
docker build -t dinov2-litserve .
docker run -p 8000:8000 dinov2-litserve
```

To Push Docker:
```bash
docker tag dinov2-litserve-api us-central1-docker.pkg.dev/moii-api-analytics/execanalytics/vertical_market:v0.0.2
gcloud auth configure-docker
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/moii-api-analytics/execanalytics/vertical_market:v0.0.2
```

