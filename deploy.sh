#!/bin/bash
# Builds the container and deploys it to Google Cloud Run

GCP_PROJECT_ID="massive-woods-500718-c8"
GCP_REGION="us-west1"
SERVICE_NAME="orcasafe-app"
IMAGE_NAME="orcasafe-api"
REPO_NAME="orcasafe-repo"

IMAGE_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

set -e

echo "Creating Artifact Registry repository..."
# Ignore error if repository already exists
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="OrcaSafe Docker Repository" \
    --project="$GCP_PROJECT_ID" || true

echo "Building container image..."
gcloud builds submit --tag "$IMAGE_URL" --project="$GCP_PROJECT_ID" .

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE_URL" \
    --platform=managed \
    --region="$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=2

echo "Deployment complete."
