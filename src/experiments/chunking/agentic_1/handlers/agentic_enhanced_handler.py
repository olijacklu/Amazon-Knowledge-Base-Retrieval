import json
import boto3
import time

MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
FALLBACK_MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"
REGION = "us-west-2"
MAX_TOKENS = 1000
TEMPERATURE = 0.2


def invoke_claude_haiku(prompt, max_retries=5):
    """Invoke Claude Haiku for context generation with retry logic"""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION)

    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    })

    for attempt in range(max_retries):
        try:
            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                contentType='application/json',
                accept='application/json',
                body=request_body
            )

            result = ""
            for event in response.get('body'):
                chunk = json.loads(event['chunk']['bytes'].decode())
                if chunk['type'] == 'content_block_delta':
                    result += chunk['delta']['text']
                elif chunk['type'] == 'message_delta':
                    if 'stop_reason' in chunk['delta']:
                        break
            
            return result.strip()
            
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                if attempt == 0:
                    print(f"Throttling detected, trying fallback model")
                    fallback_request = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                    })
                    try:
                        fallback_response = bedrock_runtime.invoke_model_with_response_stream(
                            modelId=FALLBACK_MODEL_ID,
                            contentType='application/json',
                            accept='application/json',
                            body=fallback_request
                        )
                        result = ""
                        for event in fallback_response.get('body'):
                            chunk = json.loads(event['chunk']['bytes'].decode())
                            if chunk['type'] == 'content_block_delta':
                                result += chunk['delta']['text']
                            elif chunk['type'] == 'message_delta':
                                if 'stop_reason' in chunk['delta']:
                                    break
                        return result.strip()
                    except:
                        pass
                
                print(f"Token limit hit, waiting 30s before retry {attempt + 1}/{max_retries}")
                time.sleep(30)
            else:
                print(f"Claude Haiku invocation error: {e}")
                return ""


def generate_contextual_prompt(document_content, chunk_content):
    """Generate the contextual retrieval prompt"""
    return f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


def lambda_handler(event, context):
    """Lambda handler for agentic context enhancement"""
    s3_client = boto3.client('s3')

    input_files = event.get('inputFiles')
    input_bucket = event.get('bucketName')

    if not all([input_files, input_bucket]):
        raise ValueError("Missing required input parameters")

    output_files = []
    for input_file in input_files:
        processed_batches = []
        for batch in input_file.get('contentBatches'):
            input_key = batch.get('key')
            
            if not input_key:
                raise ValueError("Missing key in content batch")
            
            try:
                response = s3_client.get_object(Bucket=input_bucket, Key=input_key)
                file_content = json.loads(response['Body'].read().decode('utf-8'))
            except Exception as e:
                print(f"Error reading file from S3: {e}")
                continue
            
            original_document_content = ''.join(
                content.get('contentBody', '') 
                for content in file_content.get('fileContents', []) 
                if content
            )
            
            chunked_content = {'fileContents': []}
            
            for content in file_content.get('fileContents', []):
                content_body = content.get('contentBody', '')
                content_type = content.get('contentType', '')
                content_metadata = content.get('contentMetadata', {})
                
                if not content_body:
                    continue
                
                chunk_text = content_body
                
                if not chunk_text.strip():
                    continue
                
                context_prompt = generate_contextual_prompt(original_document_content, chunk_text)
                chunk_context = invoke_claude_haiku(context_prompt)
                
                if chunk_context:
                    enhanced_chunk = chunk_context + "\n\n" + chunk_text
                else:
                    enhanced_chunk = chunk_text
                
                chunked_content['fileContents'].append({
                    "contentBody": enhanced_chunk,
                    "contentType": content_type,
                    "contentMetadata": content_metadata,
                })
            
            output_key = f"Output/{input_key}"
            try:
                s3_client.put_object(
                    Bucket=input_bucket,
                    Key=output_key,
                    Body=json.dumps(chunked_content),
                    ContentType='application/json'
                )
                processed_batches.append({"key": output_key})
            except Exception as e:
                print(f"Error writing output to S3: {e}")
        
        output_files.append({
            "originalFileLocation": input_file.get('originalFileLocation'),
            "fileMetadata": {},
            "contentBatches": processed_batches
        })
    
    return {"outputFiles": output_files}
