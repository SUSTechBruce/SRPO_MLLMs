import os
from dotenv import load_dotenv
load_dotenv(override=True)

ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "o4-mini")
SUBSCRIPTION_KEY = os.getenv("API_KEY")
API_VERSION = os.getenv("API_VERSION", None)
