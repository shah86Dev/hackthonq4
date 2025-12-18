from openai import OpenAI
from .settings import settings


def get_openai_client():
    """
    Create and return an OpenAI client instance
    """
    client = OpenAI(api_key=settings.openai_api_key)
    return client


# Initialize the client when module is imported
openai_client = get_openai_client()