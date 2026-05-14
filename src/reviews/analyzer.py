import os
import logging
from pathlib import Path
import torch
import numpy as np

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ReviewAnalyzer:
    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAP = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def __init__(self, model_name: str = None, cache_dir: str = "models/hf_cache"):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Nalagam HuggingFace model: {self.model_name}")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
        )

        # 2. Model (use_safetensors=True za PyTorch <2.6 kompatibilnost)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            use_safetensors=True,
        )
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Model nalozen na {self.device}")

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 1.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))

        return {
            "label": self.LABEL_MAP[predicted_idx],
            "score": float(probs_np[predicted_idx]),
            "scores": {
                self.LABEL_MAP[i]: float(probs_np[i])
                for i in range(len(self.LABEL_MAP))
            }
        }

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for j, prob in enumerate(probs_np):
                predicted_idx = int(np.argmax(prob))
                results.append({
                    "label": self.LABEL_MAP[predicted_idx],
                    "score": float(prob[predicted_idx]),
                    "scores": {
                        self.LABEL_MAP[k]: float(prob[k])
                        for k in range(len(self.LABEL_MAP))
                    }
                })

        return results


if __name__ == "__main__":
    analyzer = ReviewAnalyzer()

    test_reviews = [
        "Flight was delayed 3 hours, terrible experience.",
        "Average flight, nothing special.",
        "Amazing service! Crew was friendly and on-time.",
    ]

    print("\n" + "=" * 60)
    print("Test napovedi:")
    print("=" * 60)

    for review in test_reviews:
        result = analyzer.predict(review)
        print(f"\nText: {review}")
        print(f"  Label: {result['label']} (score: {result['score']:.3f})")
        print(f"  Scores: {result['scores']}")