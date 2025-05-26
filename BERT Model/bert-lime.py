import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer
import matplotlib.pyplot as plt

max_length = 128

class BERTWrapper:
    def __init__(self, model, tokenizer, max_length=128, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="tf"
            )

            outputs = self.model(inputs, training=False).logits
            probs = tf.nn.softmax(outputs, axis=1).numpy()
            all_probs.extend(probs)

        return np.array(all_probs)

wrapped_model = BERTWrapper(model, tokenizer, max_length=max_length, batch_size=8)

explainer = LimeTextExplainer(class_names=["Real", "Fake"])

def explain_bert_with_lime(text_to_explain, num_features=10, num_samples=300):
    explanation = explainer.explain_instance(
        text_to_explain,
        wrapped_model.predict_proba,
        num_features=num_features,
        num_samples=num_samples
    )

    probs = wrapped_model.predict_proba([text_to_explain])[0]
    pred_class = "Fake" if np.argmax(probs) == 1 else "Real"
    confidence = probs[1] if pred_class == "Fake" else probs[0]

    print(f"\nModel prediction: {pred_class} news (probability: {confidence:.4f})")
    print("Explanation (top contributing words):")
    for word, weight in explanation.as_list():
        direction = "FAKE" if weight > 0 else "REAL"
        print(f"• \"{word}\" contributes {abs(weight):.4f} toward {direction}")

    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    plt.title(f"LIME Explanation - Predicted: {pred_class}")
    plt.tight_layout()
    plt.show()

    return explanation

real_example = test_data[test_data["fake"] == 0]["text"].iloc[0]
fake_example = test_data[test_data["fake"] == 1]["text"].iloc[0]

print("===== EXPLAINING A REAL NEWS ARTICLE =====")
explain_bert_with_lime(real_example)

print("\n===== EXPLAINING A FAKE NEWS ARTICLE =====")
explain_bert_with_lime(fake_example)
