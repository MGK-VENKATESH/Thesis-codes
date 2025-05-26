import lime
print("\nExplaining predictions with LIME:")
explainer = lime.lime_text.LimeTextExplainer(class_names=["Real", "Fake"])


def predict_proba(texts):

    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)
   
    preds = model.predict(padded)
    
    return np.hstack([1-preds, preds])


def find_suitable_example(max_attempts=10):
    """Find an example with enough tokens for LIME to analyze."""
    for _ in range(max_attempts):
        idx = np.random.randint(0, len(test_data))
        text = test_data["text"].iloc[idx]
        true_label = test_data["fake"].iloc[idx]
        
        
        sequence = tokenizer.texts_to_sequences([text])[0]
        if len(sequence) >= 10:  
            return idx, text, true_label
    
   
    for idx in range(min(100, len(test_data))):
        text = test_data["text"].iloc[idx]
        true_label = test_data["fake"].iloc[idx]
        sequence = tokenizer.texts_to_sequences([text])[0]
        if len(sequence) >= 10:
            return idx, text, true_label
    
    
    return 0, test_data["text"].iloc[0], test_data["fake"].iloc[0]


idx, text, true_label = find_suitable_example()


max_words_for_lime = 1000
text_words = text.split()
if len(text_words) > max_words_for_lime:
    text = ' '.join(text_words[:max_words_for_lime])
    print(f"Note: Text was trimmed to {max_words_for_lime} words for LIME analysis")

try:
    
    exp = explainer.explain_instance(text, predict_proba, num_features=6)
    print(f"True label: {'Fake' if true_label == 1 else 'Real'}")
    print("LIME explanation:")
    for word, weight in exp.as_list():
        print(f"  {word}: {weight}")

    
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"LIME explanation failed: {e}")
    print("The text might be too short, too long, or contain words that the model doesn't recognize.")
