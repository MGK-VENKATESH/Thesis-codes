import shap

def text_to_input(texts):
    if isinstance(texts, np.ndarray):  
        texts = texts.tolist()
    elif isinstance(texts, str):
        texts = [texts]

    
    texts = [str(text) for text in texts]  
    
    sequences = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

def model_predict(texts):
    inputs = text_to_input(texts)
    return model.predict(inputs).flatten()


background_size = 50
background_indices = np.random.choice(len(test_data), background_size, replace=False)
background_texts = test_data['text'].iloc[background_indices].tolist()
background_inputs = text_to_input(background_texts)  


print("\nInitializing SHAP explainer (this may take a while)...")
explainer = shap.Explainer(model_predict, background_inputs)


n_examples = 5
sample_indices = np.random.choice(len(test_data), n_examples, replace=False)
sample_texts = test_data['text'].iloc[sample_indices].tolist()
sample_inputs = text_to_input(sample_texts)  
sample_labels = test_data['fake'].iloc[sample_indices].tolist()

print(f"Computing SHAP values for {n_examples} examples...")
shap_values = explainer(sample_inputs)


plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()


example_idx = 0
print(f"\nDetailed SHAP explanation for {'Fake' if sample_labels[example_idx] == 1 else 'Real'} news example:")
plt.figure(figsize=(12, 8))
shap.plots.waterfall(shap_values[example_idx])
plt.title(f"SHAP Explanation for {'Fake' if sample_labels[example_idx] == 1 else 'Real'} News Example")
plt.tight_layout()
plt.show()


model.save('lstm_fake_news_model.h5')
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nModel and interpretability analysis completed!")
