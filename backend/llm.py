import os
import requests

class MistralLLM:
    def __init__(self, model: str = "mistral-small-latest"):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

    def generate_response(self, query: str, context_docs: list) -> str:
        context_text = "\n\n".join([
            f"Document {i+1} (Titre: {doc.get('title', 'Sans titre')}):\n{doc.get('text', '')}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""
Tu es un assistant expert en recherche d'information. 
Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur les documents fournis dans le contexte.
Si la réponse ne se trouve pas dans les documents, dis-le clairement.
Ne fabrique pas d'informations.

### Contexte :
{context_text}

### Question de l'utilisateur :
{query}

### Réponse :
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral : {e}")
            return "Désolé, une erreur est survenue lors de la génération de la réponse."