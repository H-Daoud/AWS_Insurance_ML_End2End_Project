# tests/test_security.py

import boto3
import pytest
from moto import mock_ec2

@pytest.fixture
def ec2_client():
    # Start Moto mock EC2 service
    with mock_ec2():
        client = boto3.client("ec2", region_name="eu-central-1")
        yield client

def test_security_group_creation(ec2_client):
    # Create a VPC
    vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")
    vpc_id = vpc['Vpc']['VpcId']

    # Create the Security Group (simulate what Terraform would do)
    sg = ec2_client.create_security_group(
        GroupName="HUK-RAG-Chatbot-Service-SG",
        Description="Allows traffic to FastAPI port 8000",
        VpcId=vpc_id
    )

    sg_id = sg['GroupId']

    # Add ingress rule (mocking Terraform)
    ec2_client.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 8000,
            "ToPort": 8000,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
        }]
    )

    # Add egress rule
    ec2_client.authorize_security_group_egress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "-1",
            "FromPort": 0,
            "ToPort": 0,
            "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
        }]
    )

    # Fetch the security group
    response = ec2_client.describe_security_groups(GroupIds=[sg_id])
    sg_info = response['SecurityGroups'][0]

    # Tests
    assert sg_info['GroupName'] == "HUK-RAG-Chatbot-Service-SG"
    assert sg_info['VpcId'] == vpc_id

    # Check ingress rule
    ingress = sg_info['IpPermissions'][0]
    assert ingress['FromPort'] == 8000
    assert ingress['ToPort'] == 8000
    assert ingress['IpProtocol'] == 'tcp'
    assert ingress['IpRanges'][0]['CidrIp'] == '0.0.0.0/0'

    # Check egress rule
    egress = sg_info['IpPermissionsEgress'][0]
    assert egress['IpProtocol'] == '-1'
    assert egress['FromPort'] == 0
    assert egress['ToPort'] == 0
    assert egress['IpRanges'][0]['CidrIp'] == '0.0.0.0/0'
