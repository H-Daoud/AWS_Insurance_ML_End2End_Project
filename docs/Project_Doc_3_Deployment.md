# Project_Doc_3_Deployment_and_Operations.md

---

# NOTE: This project now uses AWS as the primary cloud infrastructure. Any references to Azure (e.g., Azure OpenAI, Azure ML, Azure-specific Terraform templates, or Azure deployment scripts) are deprecated and should be ignored for current and future deployments. All instructions, diagrams, and documentation should focus on AWS resources, workflows, and best practices.

## Deployment & Operations Instructions (30. November 2025)

### Infrastructure as Code (Terraform)
Terraform is the standard IaC tool for all cloud deployments (Azure, GCP, AWS). Each cloud folder contains a Terraform template:
- Azure: `azure/infrastructure.tf`
- GCP: `gcp/infrastructure.tf`
- AWS: `aws/infrastructure.tf`

#### General Terraform Workflow
1. Install Terraform: https://www.terraform.io/downloads.html
2. Authenticate to your cloud provider (e.g., `az login`, `gcloud auth login`, `aws configure`)
3. Initialize and apply the Terraform template:
   ```bash
   terraform init
   terraform apply
   ```
4. Update variables and credentials as needed in each template.

### 1. Local Deployment
- Run FastAPI app locally:
  ```bash
  uvicorn FastAPI_app.app:app --reload --env-file configs/dev.env
  ```
- Test endpoints at `http://localhost:8000`

### 2. Docker Deployment
- Build and run Docker image:
  ```bash
  docker build -t huk-feedback-app .
  docker run --env-file configs/dev.env -p 8000:8000 huk-feedback-app
  ```
- Test endpoints at `http://localhost:8000`

### 3. Azure ML Deployment
- Use `azure/infrastructure.tf` for resource provisioning.
- Follow the general Terraform workflow above.
- Build and push Docker image to ACR, deploy to managed endpoint (Azure ML Studio/CLI).
- Use `azure/score.py` as ONNX inference entry script.
- Test endpoint with sample requests:
  ```bash
  curl -X POST https://<endpoint-url>/score \
    -H "Content-Type: application/json" \
    -d '{"text": "The insurance claim process was fast and easy."}'
  ```

### 4. GCP Deployment
- Use `gcp/infrastructure.tf` for resource provisioning.
- Follow the general Terraform workflow above.
- Build Docker image and push to Google Container Registry (GCR):
  ```bash
  docker build -t gcr.io/<project-id>/huk-feedback-app .
  docker push gcr.io/<project-id>/huk-feedback-app
  ```
- Deploy to Google Cloud Run:
  ```bash
  gcloud run deploy huk-feedback-app \
    --image gcr.io/<project-id>/huk-feedback-app \
    --platform managed \
    --region <region> \
    --set-env-vars $(cat configs/prod.env | xargs)
  ```
- Test endpoint at provided Cloud Run URL.

### 5. AWS Deployment
- Use `aws/infrastructure.tf` for resource provisioning.
- Follow the general Terraform workflow above.
- Build Docker image and push to Amazon ECR:
  ```bash
  aws ecr create-repository --repository-name huk-feedback-app
  $(aws ecr get-login --no-include-email --region <region>)
  docker tag huk-feedback-app:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/huk-feedback-app:latest
  docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/huk-feedback-app:latest
  ```
- Deploy to AWS ECS or Lambda (via container):
  - ECS: Use Fargate launch type, set env vars from `prod.env`.
  - Lambda: Use container image, set env vars.
- Test endpoint at provided AWS URL.

---
This document covers deployment and operations for local, Docker, Azure ML, GCP, and AWS environments, all using Terraform for IaC. Update with platform-specific details as needed.
