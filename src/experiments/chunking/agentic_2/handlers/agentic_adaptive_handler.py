import json
import boto3
import time
import re

MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
REGION = "us-west-2"
MAX_TOKENS = 200
TEMPERATURE = 0.2


def invoke_claude_haiku(prompt, max_retries=5):
    """Invoke Claude Haiku for analysis with retry logic"""
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
                print(f"Token limit hit, waiting 30s before retry {attempt + 1}/{max_retries}")
                time.sleep(30)
            else:
                print(f"Claude Haiku invocation error: {e}")
                return ""


def split_into_sentences(text):
    """Split text into sentences using regex"""
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def analyze_document_structure(document_text):
    """Use LLM to analyze document structure and find breaking points"""
    sentences = split_into_sentences(document_text)
    
    word_count = len(document_text.split())
    target_chunks = round(word_count / 1000)
    min_breakpoints = max(2, target_chunks - 2)
    max_breakpoints = target_chunks + 2
    
    sentence_summary = "\n".join([f"{i}: {sentence}" for i, sentence in enumerate(sentences)])
    
    breaking_prompt = f"""You are analyzing an academic research paper to identify optimal breaking points for text chunking.

The document has {len(sentences)} total sentences and {word_count} words. Here are the sentences:

{sentence_summary}

Your task is to identify {min_breakpoints}-{max_breakpoints} optimal breaking points (sentence numbers) where the document naturally transitions between:
- Different topics or concepts
- Section boundaries
- Logical flow breaks
- Conceptual shifts
- Changes in focus or subject matter

Return ONLY the sentence numbers as a comma-separated list (e.g., "8, 15, 23, 31, 42, 55").
Do not include explanations or any other text."""

    response = invoke_claude_haiku(breaking_prompt)
    
    numbers = re.findall(r'\b\d+\b', response)
    break_points = [int(num) for num in numbers]
    
    char_positions = [0]
    current_pos = 0
    
    for i, sentence in enumerate(sentences):
        current_pos += len(sentence) + 1
        if i in break_points:
            char_positions.append(current_pos)
    
    char_positions.append(len(document_text))
    return sorted(list(set(char_positions)))


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


def create_chunks_from_breaking_points(document_text, break_points):
    """Create chunks based on LLM-identified breaking points"""
    chunks = []
    
    for i in range(len(break_points) - 1):
        start_pos = break_points[i]
        end_pos = break_points[i + 1]
        chunk_text = document_text[start_pos:end_pos].strip()
        
        chunks.append({
            'text': chunk_text,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'position': i + 1
        })
    
    return chunks


def lambda_handler(event, context):
    """Lambda handler for LLM-guided breaking points"""
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
            
            chunked_content = {'fileContents': []}
            
            for content in file_content.get('fileContents', []):
                content_body = content.get('contentBody', '')
                content_type = content.get('contentType', '')
                content_metadata = content.get('contentMetadata', {})
                
                if not content_body:
                    continue
                
                print(f"Analyzing document structure ({len(content_body)} characters)...")
                break_points = analyze_document_structure(content_body)
                print(f"Found {len(break_points) - 1} breaking points")

                chunks = create_chunks_from_breaking_points(content_body, break_points)
                print(f"Created {len(chunks)} chunks")

                for chunk_info in chunks:
                    chunk_text = chunk_info['text']
                    chunk_position = chunk_info['position']

                    context_prompt = generate_contextual_prompt(content_body, chunk_text)
                    chunk_context = invoke_claude_haiku(context_prompt)

                    if chunk_context:
                        enhanced_chunk = chunk_context + "\n\n" + chunk_text
                    else:
                        enhanced_chunk = chunk_text

                    chunked_content['fileContents'].append({
                        "contentBody": enhanced_chunk,
                        "contentType": content_type,
                        "contentMetadata": {
                            **content_metadata,
                            "chunk_position": chunk_position,
                            "original_start": chunk_info['start_pos'],
                            "original_end": chunk_info['end_pos']
                        },
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
