import requests


class QueryAugmenter:
    def __init__(self, ollama_url: str, generative_model: str, rewrite_model: str):
        self.ollama_url = ollama_url
        self.generative_model = generative_model
        self.rewrite_model = rewrite_model

    def rewrite_query(self, question: str) -> str:
        prompt = f"""You are a search query expander for document retrieval. Your job is to add synonyms and closely related terms to help find the answer in any type of document.

Rules:
- Only add terms that are directly related to what the question is asking about.
- For short factual questions ("What is X?", "Who is Y?"), keep the expansion tight — 3 to 5 terms maximum.
- Never invent terms that could belong to a different topic entirely.
- Return only the expanded query. Use English only.

Question: what is RAG
Expanded: RAG retrieval augmented generation pipeline architecture

Question: how does chunking work
Expanded: chunking text splitting document segmentation sentence splitting chunk size overlap

Question: what is the main topic
Expanded: main topic subject purpose overview summary

Question: what are the requirements
Expanded: requirements qualifications criteria conditions prerequisites

Question: how to reduce hallucination
Expanded: hallucination reduction grounding faithfulness factuality

Question: {question}
Expanded:"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.rewrite_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            expansion = response.json().get("response", "").strip()
            # Always keep original terms — expansion is additive, never a replacement.
            # This ensures named features/proper nouns from the question survive even
            # when the small rewrite model paraphrases them away.
            rewritten = f"{question} {expansion}" if expansion else question
            print(f"🔄 Query expanded: '{question}' → '{rewritten}'")
            return rewritten
        except Exception:
            return question

    def generate_hypothetical_answer(self, question: str) -> str:
        prompt = f"""Write a short, technically detailed passage (2-4 sentences) that directly answers the following question. Write as if you are the document being searched — use specific terms, model names, and technical details that would appear in a technical document.

Question: {question}
Passage:"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.generative_model, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            hypothetical = response.json().get("response", question).strip()
            print(f"💭 HyDE passage: {hypothetical[:100]}...")
            return hypothetical
        except Exception:
            return question
