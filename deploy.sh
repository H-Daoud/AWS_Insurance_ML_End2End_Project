#!/bin/bash

set -e

AWS_REGION="eu-central-1"
AWS_ACCOUNT_ID="880844766572"
REPO_NAME="huk-rag-chatbot-repo"
LOCAL_IMAGE_NAME="huk-rag-chatbot"
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "Building Docker image..."
docker build -t $LOCAL_IMAGE_NAME:latest .

echo "Tagging image..."
docker tag $LOCAL_IMAGE_NAME:latest $ECR_URL

echo "Pushing image to ECR..."
docker push $ECR_URL

echo "Done. Your container is now strolling around ECR like it owns the place."
echo "Image: $ECR_URL"

