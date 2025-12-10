# submit_training_job.py
# AWS SageMaker training job submission script

import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize SageMaker session
sess = sagemaker.Session()
role = get_execution_role()

# Define training job parameters
estimator = sagemaker.estimator.Estimator(
    image_uri='763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.0.0-cpu-py39-ubuntu20.04',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://huk-feedback-model-artifacts/',
    sagemaker_session=sess,
    hyperparameters={
        'epochs': 3,
        'learning_rate': 2e-5
    }
)

# Launch the training job using data in S3
estimator.fit({'training': 's3://huk-feedback-data/vehicle_feedback.csv'})

print("✅ SageMaker training job submitted successfully.")
