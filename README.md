🚗 **HUK-COBURG Feedback Intelligence (Prototyp)**
⚡ **A Compound AI System Hybrid ML (DistilBERT) + RAG (Azure OpenAI)**
**Pipeline:** `Local ML (Router)` ➔ `RAG (Policy Engine)` ➔ `LLM (Reasoning)`

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure)
![Model](https://img.shields.io/badge/Router-DistilBERT-yellow)
![GenAI](https://img.shields.io/badge/Reasoning-OpenAI-green?logo=openai)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![DevOps](https://img.shields.io/badge/MLOps-red)

<p align="center">
  <img src=" " width="800">
</p>
---
🚀 **Projekct overview**
To create a scalable, secure, and low-latency Rest API service that provides Insurance customers with instant, context-aware answers to policy, claim, and service questions by retrieving data from internal PDF documents.

```bash
AWS_Insurance_ML_End2End_Project/
├── aws/                  # AWS Infrastructure as Code (Terraform, CloudFormation)
│   ├── cloudformation.yaml
│   ├── ecr.tf
│   ├── ecs.tf
│   ├── infrastructure.tf
│   ├── main.tf
│   ├── network.tf
│   ├── score.py
│   ├── secrets.tf
│   ├── security.tf
│   ├── submit_training_job.py
│   ├── terraform.tfstate
│   ├── terraform.tfstate.backup
│   ├── test_score_local.py
│   ├── tfplan
│   └── versions.tf
├── configs/              # Environment Configs
│   ├── dev.env
│   └── prod.env
├── data/                 # Data Files
│   ├── processed/
│   │   ├── training_data.jsonl
│   │   ├── vector_index.faiss
│   │   └── vector_index.pkl
│   └── raw/
│       ├── insurance_terms.pdf
│       └── vehicle_feedback.csv
├── docs/                 # Documentation
│   ├── Architecture.png
│   └── Project_Doc_1_7.md
├── FastAPI_app/          # FastAPI Application
│   ├── __init__.py
│   └── app.py
├── models/               # Model Files
│   └── huk_distilbert.onnx
├── notebooks/            # Jupyter Notebooks
│   ├── 01_eda.ipynb
│   ├── 01_eda.py
├── reports/              # Reports and Figures
│   ├── system_errors.log
│   └── figures/
│       ├── class_balance.png
│       ├── confusion_matrix.png
│       ├── feedback_by_sentiment_category.png
│       ├── feedback_length_distribution.png
├── scripts/              # Utility Scripts
│   ├── ingest_data.py
│   └── setup_models.py
├── src/                  # Source Code
│   ├── __init__.py
│   ├── app.py
│   ├── exceptions.py
│   ├── logger.py
│   ├── main_api.py
│   ├── schemas.py
│   ├── utils.py
│   ├── classifier/
│   │   ├── evaluate.py
│   │   ├── export_onnx.py
│   │   ├── inference.py
│   │   └── train.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── engine.py
│   │   └── vector_store.py
│   │   └── pii_scrubber.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── tests/                # Test Suite
│   ├── conftest.py
│   ├── test_classifier.py
│   ├── test_rag.py
│   └── test_security.py
├── Security/                # maintenance suite
│   ├──  __init__.py
│   ├── pii_scrubber.py
│   ├──auth.py
├── score.py    
├── Dockerfile            # Docker Build File
├── README.md             # Project Documentation
└── requirements.txt      # Python Dependencies
└── .dockerignore      # Python Dependencies

```bash
# 1. Remove the local Python environment (often the largest culprit)
rm -rf .venv

# 2. Remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf logs/
rm -rf results/
## 🛠️ Step-by-Step: Build & Deploy on AWS
1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd CC_HD_aws
   ```
2. **Prepare Your Environment**
   - Copy and edit environment files:
     ```bash
     cp configs/dev.env configs/prod.env
     # Edit prod.env with your AWS secrets, keys, and config
     ```

3. **Build the Docker Image**
   - Exclude large files and folders (see .gitignore):
     - `data/` (not needed in Docker if you already have the trained ONNX model)
     - `models/*.onnx` (only include if needed for inference)
     - `reports/`, `results/`, `.venv/`, `__pycache__/`, `*.log`, `*.csv`, `*.pdf`
   - Build the image:
     ```bash
     docker build -t huk-feedback-app .
     ```

4. **Push Docker Image to AWS ECR**
   - Create ECR repository:
     ```bash
     aws ecr create-repository --repository-name huk-feedback-app
     ```
   - Authenticate Docker to ECR:
     ```bash
     aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
     ```
   - Tag and push:
     ```bash
     docker tag huk-feedback-app:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/huk-feedback-app:latest
     docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/huk-feedback-app:latest
     ```

5. **Deploy on AWS ECS (Fargate)**
   - Use Terraform or CloudFormation templates in `aws/` to provision ECS Cluster, Task Definition, Service, and Security Group.

6. **Get the Public URL**
   - After deployment, find the Load Balancer DNS name or Service Public IP in the AWS ECS Console.
   - The app runs on port `8000`, so your public URL will look like:
     ```
     http://<your-public-dns>:8000
     ```
   - Test with:
     ```bash
     curl http://<your-public-dns>:8000/docs
     ```
---

## 🧩 How the Codebase Fits Together & Design Rationale

This project is architected for real-world production, with each component chosen for a specific reason. Below is an explanation of how the files and modules connect, and why each approach was selected, referencing the ASCII diagram above.

### 1. Security
- **src/security/**: Contains `auth.py` (API key middleware) and `pii_scrubber.py` (removes sensitive data before cloud transfer).
- **Reason**: Protects user data, prevents unauthorized access, and ensures GDPR compliance.

### 2. DevOps & Automation
- **.github/workflows/**: CI/CD pipeline for linting, security scanning, and unit tests.
- **Makefile**: Automates common tasks (train, run, deploy).
- **aws/**: Infrastructure as Code (Terraform, CloudFormation) for reproducible, automated cloud deployments.
- **Reason**: Enables fast, reliable deployments and easy rollback; reduces manual errors.

### 3. Latency & Performance
- **models/huk_distilbert.onnx**: ONNX format for fast, CPU-optimized inference.
- **src/classifier/**: Handles local ML routing for low-latency predictions.
- **src/rag/vector_store.py**: Uses FAISS for fast vector search.
- **Reason**: Minimizes response time for user queries and optimizes resource usage.

### 4. Cost Efficiency
- **src/rag/cache.py**: Semantic caching to avoid repeated expensive OpenAI API calls.
- **ONNX model**: Reduces cloud compute costs by using efficient inference.
- **Reason**: Keeps cloud costs predictable and low, especially at scale.

### 5. Scalability
- **aws/ecs.tf, infrastructure.tf**: ECS Fargate for container orchestration and auto-scaling.
- **src/**: Modular design allows horizontal scaling of API and ML components.
- **Reason**: Supports growth in user traffic and data volume without major redesign.

### 6. Production Readiness
- **src/main_api.py**: FastAPI backend with health checks and request IDs for traceability.
- **configs/prod.env**: Environment separation for secure production deployments.
- **Reason**: Ensures reliability, observability, and maintainability in production.

### 7. Testing & Evaluation
- **tests/**: Pytest-based unit and integration tests for classifier, RAG, and security modules.
- **reports/figures/**: Visualizations (confusion matrix, class balance) for model evaluation.
- **Reason**: Guarantees code quality and model accuracy before deployment.

### 8. Monitoring & Observability
- **src/utils/logger.py**: Centralized JSON logging for performance and error tracking.
- **AWS CloudWatch**: (enabled via ECS) for real-time monitoring and alerting.
- **Reason**: Enables proactive issue detection and system health monitoring.

### 9. User Experience & Accessibility
- **FastAPI auto-generated docs**: Accessible at `/docs` for easy API exploration.
- **Reason**: Makes the system easy to use and test for both developers and stakeholders.
- **Streamlit_/app.py**: Interactive frontend for demo and visualization.

-

**Summary:**
- Every file and module is placed for a reason: security, speed, cost, scalability, and maintainability.
- The architecture supports rapid development, robust production deployment, and easy monitoring/testing.
- The modular design means you can swap out components (e.g., ML model, vector store, cloud provider) with minimal changes.

👨‍💻 Autor
Hassan Daoud