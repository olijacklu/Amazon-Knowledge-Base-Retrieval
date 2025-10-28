REGION = 'us-west-2'
BUCKET_NAME = 'arpc-converted'

CHUNKING_CONFIG = {
    "chunkingStrategy": "HIERARCHICAL",
    "hierarchicalChunkingConfiguration": {
        "levelConfigurations": [{"maxTokens": 2000}, {"maxTokens": 500}],
        "overlapTokens": 100
    }
}

GRAPH_METHOD = "hierarchical-2000-500-100"
