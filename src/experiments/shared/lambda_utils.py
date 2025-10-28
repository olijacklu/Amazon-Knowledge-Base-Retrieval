import json
import time
import zipfile
import io
import boto3


def create_lambda_function(function_name, handler_path, description, region='us-west-2'):
    """Create or update Lambda function from handler file"""
    lambda_client = boto3.client('lambda', region_name=region)
    iam_client = boto3.client('iam', region_name=region)
    sts_client = boto3.client('sts')
    
    role_name = f"{function_name}-role"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        try:
            iam_client.delete_role_policy(RoleName=role_name, PolicyName=f"{function_name}-policy")
        except:
            pass
        try:
            iam_client.detach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
        except:
            pass
        try:
            iam_client.delete_role(RoleName=role_name)
            print(f"Deleted existing IAM role: {role_name}")
            time.sleep(5)
        except:
            pass
    except:
        pass
    
    role_response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Execution role for {function_name}"
    )
    role_arn = role_response['Role']['Arn']
    
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
    )
    
    account_id = sts_client.get_caller_identity()['Account']
    custom_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:*",
                "Resource": [
                    "arn:aws:s3:::arpc-converted",
                    "arn:aws:s3:::arpc-converted/*",
                    "arn:aws:s3:::arpc-intermediate",
                    "arn:aws:s3:::arpc-intermediate/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModelWithResponseStream"],
                "Resource": "*"
            }
        ]
    }
    
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName=f"{function_name}-policy",
        PolicyDocument=json.dumps(custom_policy)
    )
    
    time.sleep(10)
    
    with open(handler_path, 'r') as f:
        lambda_code = f.read()
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('lambda_function.py', lambda_code)
    
    zip_buffer.seek(0)
    
    try:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.12',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_buffer.read()},
            Description=description,
            Timeout=900,
            MemorySize=1024
        )
        function_arn = response['FunctionArn']
        print(f"Created Lambda function: {function_name}")
    except lambda_client.exceptions.ResourceConflictException:
        zip_buffer.seek(0)
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_buffer.read()
        )
        response = lambda_client.get_function(FunctionName=function_name)
        function_arn = response['Configuration']['FunctionArn']
        print(f"Updated Lambda function: {function_name}")
    
    return function_arn
