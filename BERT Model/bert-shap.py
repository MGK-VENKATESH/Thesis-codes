import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import matplotlib.pyplot as plt
import shap
from matplotlib.patches import Patch

model_path = "bert_fake_news_model"
test_data = pd.read_csv("test_data.csv")
print(f"Loaded test data with {len(test_data)} examples")

model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
print(f"Successfully loaded model and tokenizer from {model_path}")

max_length = 128

def model_predict(texts):
    if isinstance(texts, str):
        texts = [texts]

    encoded_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    outputs = model(encoded_inputs, training=False).logits
    probs = tf.nn.softmax(outputs, axis=1).numpy()
    return probs

masker = shap.maskers.Text(tokenizer=None)
print("\nInitializing SHAP Partition explainer...")
explainer = shap.explainers.Partition(model_predict, masker)

n_examples = 3
sample_indices = np.random.choice(len(test_data), n_examples, replace=False)
sample_texts = test_data['text'].iloc[sample_indices].tolist()
sample_labels = test_data['fake'].iloc[sample_indices].tolist()

shortened_samples = []
for text in sample_texts:
    words = text.split()[:50]
    shortened_samples.append(" ".join(words))

print(f"Computing SHAP values for {n_examples} examples...")

word_importance = {}

for i, text in enumerate(shortened_samples):
    print(f"\nAnalyzing example {i+1}/{n_examples}...")
    words = text.split()
    baseline_pred = model_predict(text)[0, 1]
    word_values = []

    for j, word in enumerate(words):
        masked_words = words.copy()
        masked_words[j] = "[MASK]"
        masked_text = " ".join(masked_words)
        masked_pred = model_predict(masked_text)[0, 1]
        importance = baseline_pred - masked_pred
        word_values.append(importance)

        if word in word_importance:
            word_importance[word].append(importance)
        else:
            word_importance[word] = [importance]

    pred_proba = model_predict(text)[0, 1]
    pred_class = "Fake" if pred_proba > 0.5 else "Real"
    true_class = "Fake" if sample_labels[i] == 1 else "Real"

    print(f"Example {i+1}: '{text[:100]}...'")
    print(f"True class: {true_class}, Predicted: {pred_class} (probability: {pred_proba:.4f})")

    sorted_importance = sorted(zip(words, word_values), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 influential words for example {i+1}:")
    for word, value in sorted_importance[:10]:
        impact = "→ More Fake" if value > 0 else "→ More Real"
        print(f"'{word}': {value:.4f} {impact}")

    plt.figure(figsize=(12, 6))
    top_words = sorted_importance[:15]
    words_to_plot = [word for word, _ in top_words]
    values_to_plot = [value for _, value in top_words]

    bars = plt.bar(
        range(len(words_to_plot)),
        values_to_plot,
        color=[('red' if v > 0 else 'blue') for v in values_to_plot]
    )

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(range(len(words_to_plot)), words_to_plot, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Importance Value')
    plt.title(f'Word Importance for {true_class} News Example (Predicted: {pred_class})')

    legend_elements = [
        Patch(facecolor='red', label='Indicates Fake News'),
        Patch(facecolor='blue', label='Indicates Real News')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(f"word_importance_example_{i+1}.png")
    plt.show()

avg_word_importance = {}
for word, values in word_importance.items():
    avg_word_importance[word] = sum(values) / len(values)

top_words = sorted(avg_word_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

print("\nTop 20 words by average importance across all examples:")
for word, value in top_words:
    impact = "→ Indicates FAKE news" if value > 0 else "→ Indicates REAL news"
    print(f"'{word}': {value:.4f} {impact}")

plt.figure(figsize=(14, 8))
words_to_plot = [word for word, _ in top_words]
values_to_plot = [value for _, value in top_words]

bars = plt.barh(
    range(len(words_to_plot)),
    values_to_plot,
    color=[('red' if v > 0 else 'blue') for v in values_to_plot]
)

plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.yticks(range(len(words_to_plot)), words_to_plot)
plt.xlabel('Average Importance Value')
plt.ylabel('Words')
plt.title('Global Word Importance for Fake News Detection')

legend_elements = [
    Patch(facecolor='red', label='Indicates Fake News'),
    Patch(facecolor='blue', label='Indicates Real News')
]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("global_word_importance.png")
plt.show()

print("\nModel interpretability analysis completed!")
