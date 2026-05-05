import requests


class AnswerGenerator:
    def __init__(self, ollama_url: str, generative_model: str):
        self.ollama_url = ollama_url
        self.generative_model = generative_model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.generative_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
