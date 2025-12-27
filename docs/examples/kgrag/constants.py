import os
from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE", ".env")
dotenv_path = os.path.join(os.path.dirname(__file__), env_file)
load_dotenv(dotenv_path)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD=os.environ.get("NEO4J_PASSWORD", "")

DATASET_PATH = os.environ.get("KG_BASE_DIRECTORY", os.path.dirname(os.path.abspath(__file__)) + "/data")

# Evaluation

API_KEY="dummy"
API_BASE="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"
MODEL_NAME="meta-llama/llama-3-3-70b-instruct"
RITS_API_KEY=os.environ.get("RITS_API_KEY", "")
CONTEXT_LENGTH=131072


API_KEY="dummy"
API_BASE="http://localhost:7878/v1"
MODEL_NAME="/net/storage149/autofs/css22/nmg/models/hf/meta-llama/Llama-3.1-8B-Instruct/main"

MAX_RETRIES=3
TIME_OUT=1800

# Embedding
EMB_API_KEY="dummy"
EMB_API_BASE="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2/v1"
EMB_MODEL_NAME="ibm/slate-125m-english-rtrvr-v2"

EMB_TIME_OUT=TIME_OUT
EMB_CONTEXT_LENGTH=512

# Evaluation
EVAL_API_KEY="dummy"
EVAL_API_BASE="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"
EVAL_MODEL_NAME="meta-llama/llama-3-3-70b-instruct"

EVAL_TIME_OUT=TIME_OUT

