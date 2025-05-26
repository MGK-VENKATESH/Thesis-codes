import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix



negative_df = pd.read_csv("/kaggle/input/fake-news-detection/true.csv")
negative_df["fake"] = 0  
positive_df = pd.read_csv("/kaggle/input/fake-news-detection/fake.csv")
positive_df["fake"] = 1  

train_df = pd.concat([negative_df, positive_df]).sample(frac=1, random_state=42)  

train_data, temp_data = train_test_split(train_df, test_size=0.4, stratify=train_df["fake"], random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data["fake"], random_state=42)


print(f"Train Data: {train_data.shape}, Validation Data: {valid_data.shape}, Test Data: {test_data.shape}")


def create_dataset(dataframe, shuffle=True, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["text"].values, dataframe["fake"].values))
    if shuffle:
        dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_data)
valid_ds = create_dataset(valid_data, shuffle=False)
test_ds = create_dataset(test_data, shuffle=False)


max_words = 10000  
max_sequence_length = 100  

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["text"])

def tokenize_data(dataframe):
    sequences = tokenizer.texts_to_sequences(dataframe["text"])
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)


X_train_pad = tokenize_data(train_data)
X_valid_pad = tokenize_data(valid_data)
X_test_pad = tokenize_data(test_data)
y_train = train_data["fake"].values
y_valid = valid_data["fake"].values
y_test = test_data["fake"].values


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128),
    tf.keras.layers.SpatialDropout1D(0.2),  
    tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),  
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    X_train_pad, y_train, 
    epochs=20, 
    batch_size=64, 
    validation_data=(X_valid_pad, y_valid),
    callbacks=[early_stopping]
)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


y_pred = model.predict(X_test_pad)
y_pred_binary = (y_pred > 0.5).astype(int) 


accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(y_test, y_pred_binary)
accuracy_value = accuracy.result().numpy()

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f'Accuracy: {accuracy_value * 100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred_binary, target_names=["Real", "Fake"]))


cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
