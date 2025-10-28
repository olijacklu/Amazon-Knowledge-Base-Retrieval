import time
import boto3


def create_kb_with_lambda(chunking_config, lambda_arn, kb_name, index_name, collection_arn, region='us-west-2'):
    """Create KB with custom Lambda transformation"""
    bedrock_agent_client = boto3.client('bedrock-agent', region_name=region)
    sts_client = boto3.client('sts')
    
    account_id = sts_client.get_caller_identity()['Account']
    role_arn = f'arn:aws:iam::{account_id}:role/space-knowledge-base'
    
    existing_kbs = bedrock_agent_client.list_knowledge_bases()
    existing_kb_id = None
    for kb in existing_kbs['knowledgeBaseSummaries']:
        if kb['name'] == kb_name:
            existing_kb_id = kb['knowledgeBaseId']
            print(f"Knowledge Base {kb_name} already exists: {existing_kb_id}")
            break
    
    if existing_kb_id:
        return existing_kb_id
    
    kb = bedrock_agent_client.create_knowledge_base(
        name=kb_name,
        description=f"KB with {kb_name}",
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
            }
        },
        storageConfiguration={
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": collection_arn,
                "vectorIndexName": index_name,
                "fieldMapping": {
                    "vectorField": "vector",
                    "textField": "text",
                    "metadataField": "text-metadata"
                }
            }
        }
    )
    
    kb_id = kb['knowledgeBase']['knowledgeBaseId']
    
    time.sleep(30)
    
    ds = bedrock_agent_client.create_data_source(
        name=f"ds-{kb_name}",
        knowledgeBaseId=kb_id,
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": "arn:aws:s3:::arpc-converted",
                "inclusionPrefixes": ["md/"]
            }
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": chunking_config,
            "customTransformationConfiguration": {
                "intermediateStorage": {
                    "s3Location": {
                        "uri": "s3://arpc-intermediate/"
                    }
                },
                "transformations": [{
                    "stepToApply": "POST_CHUNKING",
                    "transformationFunction": {
                        "transformationLambdaConfiguration": {
                            "lambdaArn": lambda_arn
                        }
                    }
                }]
            }
        }
    )
    
    ds_id = ds['dataSource']['dataSourceId']
    print(f"Created data source with Lambda: {ds_id}")
    
    print(f"Starting ingestion for {kb_name}...")
    job = bedrock_agent_client.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job['ingestionJob']['ingestionJobId']
    
    max_wait_time = 1800
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        job_status = bedrock_agent_client.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )['ingestionJob']
        
        status = job_status['status']
        print(f"Ingestion status: {status}")
        
        if status == 'COMPLETE':
            stats = job_status['statistics']
            print(f"Ingestion complete: {stats['numberOfNewDocumentsIndexed']} documents indexed")
            break
        elif status == 'FAILED':
            print("Ingestion failed!")
            raise Exception("Ingestion job failed")
        
        time.sleep(60)
    
    print(f"Knowledge Base created: {kb_id}")
    return kb_id
