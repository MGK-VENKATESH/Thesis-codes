import lime
import lime.lime_text
class LSTMWrapper:
    def __init__(self, model, tokenizer, max_sequence_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        
    def predict_proba(self, texts):
        
        if isinstance(texts, str):
            texts = [texts]
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_sequence_length)
        predictions = self.model.predict(padded)
        
        
        return np.hstack([1-predictions, predictions])


wrapper = LSTMWrapper(model, tokenizer, max_sequence_length)


explainer = lime.lime_text.LimeTextExplainer(class_names=['Real', 'Fake'])


def explain_with_lime(text_to_explain, num_features=10):
    explanation = explainer.explain_instance(
        text_to_explain, 
        wrapper.predict_proba, 
        num_features=num_features
    )
    
    
    prob = wrapper.predict_proba([text_to_explain])[0, 1]
    pred_class = "Fake" if prob > 0.5 else "Real"
    
    print(f"Model prediction: {pred_class} news (probability: {prob:.4f})")
    print("\nExplanation (words contributing to the prediction):")
    
    
    for feature, weight in explanation.as_list():
        direction = "FAKE" if weight > 0 else "REAL"
        print(f"• \"{feature}\": {abs(weight):.4f} points toward {direction}")
    
    
    plt.figure(figsize=(10, 6))
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
    
    return explanation


real_news_example = test_data[test_data['fake'] == 0]['text'].iloc[5]
fake_news_example = test_data[test_data['fake'] == 1]['text'].iloc[5]

print("===== EXPLAINING A REAL NEWS ARTICLE =====")
real_explanation = explain_with_lime(real_news_example)

print("\n===== EXPLAINING A FAKE NEWS ARTICLE =====")
fake_explanation = explain_with_lime(fake_news_example)
