import json
import time
import boto3


REGION = 'us-west-2'
BUCKET_NAME = 'arpc-converted'

bedrock_agent_client = boto3.client('bedrock-agent', region_name=REGION)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name=REGION)
aoss_client = boto3.client('opensearchserverless', region_name=REGION)
sts_client = boto3.client('sts')

SHARED_COLLECTION_ID = None
SHARED_COLLECTION_ARN = None


def get_role_arn():
    """Get the IAM role ARN for Knowledge Base"""
    account_id = sts_client.get_caller_identity()['Account']
    return f'arn:aws:iam::{account_id}:role/space-knowledge-base'


def setup_shared_infrastructure():
    """Setup shared OpenSearch collection and policies"""
    global SHARED_COLLECTION_ID, SHARED_COLLECTION_ARN
    
    collection_name = 'kb-shared-collection'
    current_user_arn = sts_client.get_caller_identity()['Arn']
    role_arn = get_role_arn()
    
    print("Setting up shared infrastructure...")
    
    try:
        aoss_client.create_security_policy(
            name='kb-shared-encryption',
            policy=json.dumps({
                "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{collection_name}"]}],
                "AWSOwnedKey": True
            }),
            type='encryption'
        )
    except aoss_client.exceptions.ConflictException:
        pass
    
    try:
        aoss_client.create_security_policy(
            name='kb-shared-network',
            policy=json.dumps([{
                "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{collection_name}"]}],
                "AllowFromPublic": True
            }]),
            type='network'
        )
    except aoss_client.exceptions.ConflictException:
        pass
    
    try:
        aoss_client.create_access_policy(
            name='kb-shared-access',
            policy=json.dumps([{
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"], 
                        "Permission": ["aoss:CreateCollectionItems", "aoss:DeleteCollectionItems", 
                                     "aoss:UpdateCollectionItems", "aoss:DescribeCollectionItems"]
                    },
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{collection_name}/*"], 
                        "Permission": ["aoss:CreateIndex", "aoss:DeleteIndex", "aoss:UpdateIndex",
                                     "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument"]
                    }
                ],
                "Principal": [role_arn, current_user_arn]
            }]),
            type='data'
        )
    except aoss_client.exceptions.ConflictException:
        pass
    
    existing_collections = aoss_client.list_collections(collectionFilters={'name': collection_name})
    
    if existing_collections['collectionSummaries']:
        SHARED_COLLECTION_ID = existing_collections['collectionSummaries'][0]['id']
        SHARED_COLLECTION_ARN = existing_collections['collectionSummaries'][0]['arn']
        print(f"Using existing collection: {SHARED_COLLECTION_ID}")
    else:
        collection = aoss_client.create_collection(name=collection_name, type='VECTORSEARCH')
        SHARED_COLLECTION_ID = collection['createCollectionDetail']['id']
        SHARED_COLLECTION_ARN = collection['createCollectionDetail']['arn']
        print(f"Created new collection: {SHARED_COLLECTION_ID}")
    
    time.sleep(30)
    print("Shared infrastructure ready")
    return SHARED_COLLECTION_ID


def get_chunking_name(chunking_config):
    """Generate name from chunking configuration"""
    strategy = chunking_config["chunkingStrategy"]
    
    if strategy == "FIXED_SIZE":
        tokens = chunking_config["fixedSizeChunkingConfiguration"]["maxTokens"]
        overlap = chunking_config["fixedSizeChunkingConfiguration"]["overlapPercentage"]
        return f"fixed-{tokens}-{overlap}"
    
    elif strategy == "HIERARCHICAL":
        levels = chunking_config["hierarchicalChunkingConfiguration"]["levelConfigurations"]
        overlap = chunking_config["hierarchicalChunkingConfiguration"]["overlapTokens"]
        level_tokens = "-".join([str(level["maxTokens"]) for level in levels])
        return f"hierarchical-{level_tokens}-{overlap}"
    
    elif strategy == "SEMANTIC":
        threshold = chunking_config["semanticChunkingConfiguration"]["breakpointPercentileThreshold"]
        buffer = chunking_config["semanticChunkingConfiguration"]["bufferSize"]
        max_tokens = chunking_config["semanticChunkingConfiguration"]["maxTokens"]
        return f"semantic-{threshold}-{buffer}-{max_tokens}"
    
    return "unknown"


def wait_for_kb_ready(kb_id, max_wait=300):
    """Wait for Knowledge Base to be ready"""
    print(f"Waiting for KB {kb_id} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        kb_details = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
        status = kb_details['knowledgeBase']['status']
        
        print(f"KB status: {status}")
        
        if status == 'ACTIVE':
            print("KB is ready!")
            return True
        elif status == 'FAILED':
            print("KB creation failed!")
            return False
        
        time.sleep(10)
    
    print("Timeout waiting for KB")
    return False


def create_kb_with_chunking(chunking_config):
    """Create a Knowledge Base with specific chunking configuration"""
    if not SHARED_COLLECTION_ID:
        raise Exception("Must call setup_shared_infrastructure() first")
    
    chunking_name = get_chunking_name(chunking_config)
    kb_name = f"kb-{chunking_name}"
    index_name = f"idx-{chunking_name}"
    role_arn = get_role_arn()
    
    print(f"Creating Knowledge Base: {kb_name}")
    
    existing_kbs = bedrock_agent_client.list_knowledge_bases()
    existing_kb_id = None
    for kb in existing_kbs['knowledgeBaseSummaries']:
        if kb['name'] == kb_name:
            existing_kb_id = kb['knowledgeBaseId']
            print(f"Knowledge Base {kb_name} already exists: {existing_kb_id}")
            break
    
    if existing_kb_id:
        kb_id = existing_kb_id
        print(f"Using existing KB: {kb_id}")
    else:
        kb = bedrock_agent_client.create_knowledge_base(
            name=kb_name,
            description=f"KB with {chunking_name} chunking",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0"
                }
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": SHARED_COLLECTION_ARN,
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
        
        if not wait_for_kb_ready(kb_id):
            raise Exception(f"KB {kb_id} failed to become ready")
    
    if existing_kb_id:
        try:
            ds_response = bedrock_agent_client.list_data_sources(knowledgeBaseId=kb_id)
            if ds_response['dataSourceSummaries']:
                ds_id = ds_response['dataSourceSummaries'][0]['dataSourceId']
                print(f"Using existing data source: {ds_id}")
            else:
                ds = bedrock_agent_client.create_data_source(
                    name=f"ds-{chunking_name}",
                    knowledgeBaseId=kb_id,
                    dataSourceConfiguration={
                        "type": "S3",
                        "s3Configuration": {
                            "bucketArn": f"arn:aws:s3:::{BUCKET_NAME}",
                            "inclusionPrefixes": ["md/"]
                        }
                    },
                    vectorIngestionConfiguration={
                        "chunkingConfiguration": chunking_config
                    }
                )
                ds_id = ds['dataSource']['dataSourceId']
                print(f"Created data source: {ds_id}")
        except Exception as e:
            print(f"Error checking data sources: {e}")
            ds = bedrock_agent_client.create_data_source(
                name=f"ds-{chunking_name}",
                knowledgeBaseId=kb_id,
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": {
                        "bucketArn": f"arn:aws:s3:::{BUCKET_NAME}",
                        "inclusionPrefixes": ["md/"]
                    }
                },
                vectorIngestionConfiguration={
                    "chunkingConfiguration": chunking_config
                }
            )
            ds_id = ds['dataSource']['dataSourceId']
            print(f"Created data source: {ds_id}")
    else:
        ds = bedrock_agent_client.create_data_source(
            name=f"ds-{chunking_name}",
            knowledgeBaseId=kb_id,
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{BUCKET_NAME}",
                    "inclusionPrefixes": ["md/"]
                }
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": chunking_config
            }
        )
        ds_id = ds['dataSource']['dataSourceId']
        print(f"Created data source: {ds_id}")
    
    print(f"Starting fresh ingestion for {kb_name}...")
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
            if stats['numberOfNewDocumentsIndexed'] == 0:
                print("WARNING: No documents were indexed!")
            break
        elif status == 'FAILED':
            print("Ingestion failed!")
            if 'failureReasons' in job_status:
                print(f"Failure reasons: {job_status['failureReasons']}")
            raise Exception("Ingestion job failed")
        
        time.sleep(60)
    
    if time.time() - start_time >= max_wait_time:
        print("Ingestion timeout!")
        raise Exception("Ingestion job timed out")
    
    print(f"Knowledge Base created: {kb_id}")
    return kb_id
