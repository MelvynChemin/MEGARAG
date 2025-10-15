import os
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env and verify they exist."""
    load_dotenv()

    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
    langchain_project = os.getenv("LANGCHAIN_PROJECT")
    langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    user_agent = os.getenv("USER_AGENT")
    if langchain_api_key:
        print("✅ Environment loaded successfully (LANGCHAIN_API_KEY found)")
    else:
        print("⚠️ Warning: LANGCHAIN_API_KEY not found in environment")

    # Return values as a dict for convenience
    return {
        "LANGCHAIN_API_KEY": langchain_api_key,
        "LANGCHAIN_TRACING_V2": langchain_tracing,
        "LANGCHAIN_PROJECT": langchain_project,
        "LANGCHAIN_ENDPOINT": langchain_endpoint,
        "USER_AGENT": user_agent,
    }

def test_langsmith_connection(project_name="my-project", description="RAG testing project"):
    """Test connection to LangSmith and ensure project exists/created."""
    try:
        from langsmith import Client
        client = Client()
        print("✅ LangSmith client connected successfully")

        try:
            project = client.create_project(
                project_name=project_name,
                description=description
            )
            print(f"✅ Project created: {project.name}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"✅ Project already exists: {project_name}")
            else:
                print(f"⚠️ Project creation issue: {e}")

    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")