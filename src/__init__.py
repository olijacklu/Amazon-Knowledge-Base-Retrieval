from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
