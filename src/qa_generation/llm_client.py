import json
import boto3

MODEL_ID = "us.anthropic.claude-opus-4-1-20250805-v1:0"
REGION = "us-west-2"
MAX_TOKENS = 1000
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You generate BOTH extractive QAs and salient entities for multi-hop QA from ONE passage.
Rules:
- Produce up to N extractive QAs: answers must be EXACT substrings of the passage with 0-based char offsets.
- Also extract up to 5 salient entities/terms (datasets, models, tasks, metrics, distinctive concepts). Short, canonical, deduplicated.
Return STRICT JSON only:
{
  "entities": ["e1","e2","e3","e4","e5"],
  "qas": [
    {"question":"...?",
     "answer":"<exact substring>",
     "answer_span":{"start_char":INT,"end_char":INT}
    }
  ]
}"""

MULTIHOP_SYSTEM_PROMPT = """You generate MULTI-HOP extractive QAs that REQUIRE both passages.
Rules:
- Create up to N questions that cannot be answered from only one passage.
- The answer must be an EXACT substring from ONE passage with 0-based char offsets.
Return STRICT JSON only:
{"qas":[
  {"question":"...?",
   "answer":"<exact substring>",
   "answer_span":{"passage":"A"|"B","start_char":INT,"end_char":INT}
  }
]}"""


def build_single_hop_prompt(passage, max_qas, max_entities=5):
    """Build prompt for single-hop QA generation"""
    return f"""Max QAs: {max_qas}
Max Entities: {max_entities}

PASSAGE:
<<<
{passage}
>>>

Return STRICT JSON exactly as specified."""


def build_multihop_prompt(passage_a, passage_b, max_qas=1, paper_a=None, paper_b=None):
    """Build prompt for multi-hop QA generation"""
    paper_info = ""
    if paper_a and paper_b:
        paper_info = f"PAPER A: {paper_a}\n"
    
    return f"""Max QAs: {max_qas}

{paper_info}PASSAGE A:
<<<
{passage_a}
>>>

{f"PAPER B: {paper_b}" if paper_b else ""}
PASSAGE B:
<<<
{passage_b}
>>>

Return STRICT JSON exactly as specified."""


def extract_json(response):
    """Extract JSON from LLM response"""
    response = response.strip()
    
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    if "{" in response and "}" in response:
        start = response.find("{")
        brace_count = 0
        end = start
        for i, char in enumerate(response[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        return response[start:end].strip()
    
    return response.strip()


def call_bedrock_claude(system_prompt, user_prompt):
    """Call AWS Bedrock Claude API"""
    client = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
    }

    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())

    try:
        return payload["output"]["message"]["content"][0]["text"]
    except Exception:
        try:
            return payload["message"]["content"][0]["text"]
        except Exception:
            return payload.get("content", [{}])[0].get("text")
