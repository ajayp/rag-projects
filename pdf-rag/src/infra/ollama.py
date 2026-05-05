import json
import requests


def setup_ollama_models(
    ollama_url: str,
    embedding_model: str,
    generative_model: str,
    rewrite_model: str,
) -> None:
    print("Setting up Ollama models...")
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]

            if not any(embedding_model in m for m in models):
                print(f"Pulling embedding model: {embedding_model}")
                _pull_ollama_model(ollama_url, embedding_model)

            if not any(generative_model in m for m in models):
                print(f"Pulling generative model: {generative_model}")
                _pull_ollama_model(ollama_url, generative_model)

            if not any(rewrite_model in m for m in models):
                print(f"Pulling rewrite model: {rewrite_model}")
                _pull_ollama_model(ollama_url, rewrite_model)

            print("✅ Ollama models ready!")
        else:
            print("⚠️  Could not connect to Ollama. Make sure it's running on port 11434")
    except Exception as e:
        print(f"⚠️  Ollama setup error: {e}")


def _pull_ollama_model(ollama_url: str, model_name: str) -> None:
    try:
        response = requests.post(
            f"{ollama_url}/api/pull",
            json={"name": model_name},
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "status" in data:
                    print(f"  {data['status']}")
                if data.get("status") == "success":
                    break
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")
