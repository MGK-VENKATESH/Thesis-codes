import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, TFBertForSequenceClassification, InputExample, InputFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

fake_df = pd.read_csv("/content/Fake.csv")
true_df = pd.read_csv("/content/True.csv")

fake_df['fake'] = 1
true_df['fake'] = 0

df = pd.concat([fake_df[['text', 'fake']], true_df[['text', 'fake']]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fake'])
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['fake'])

train_data.to_csv("train_data.csv", index=False)
valid_data.to_csv("valid_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

train_data = pd.read_csv("train_data.csv")
valid_data = pd.read_csv("valid_data.csv")
test_data = pd.read_csv("test_data.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def convert_data_to_examples(train, validation, test):
    train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                             text_a=str(x['text']),
                                                             label=int(x['fake'])), axis=1)
    val_InputExamples = validation.apply(lambda x: InputExample(guid=None,
                                                                text_a=str(x['text']),
                                                                label=int(x['fake'])), axis=1)
    test_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                           text_a=str(x['text']),
                                                           label=int(x['fake'])), axis=1)
    return train_InputExamples, val_InputExamples, test_InputExamples

def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []
    for e in examples:
        inputs = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        features.append(
            InputFeatures(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=[0] * max_length,
                label=e.label
            )
        )

    def gen():
        for f in features:
            yield ({
                'input_ids': f.input_ids,
                'attention_mask': f.attention_mask,
                'token_type_ids': f.token_type_ids
            }, f.label)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'token_type_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    )

train_examples, val_examples, test_examples = convert_data_to_examples(train_data, valid_data, test_data)

train_dataset = convert_examples_to_tf_dataset(train_examples, tokenizer).shuffle(100).batch(32).repeat()
val_dataset = convert_examples_to_tf_dataset(val_examples, tokenizer).batch(32)
test_dataset = convert_examples_to_tf_dataset(test_examples, tokenizer).batch(32)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

batch_size = 32
train_steps = (len(train_data) + batch_size - 1) // batch_size
val_steps = (len(valid_data) + batch_size - 1) // batch_size
test_steps = (len(test_data) + batch_size - 1) // batch_size

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=20,
                    steps_per_epoch=train_steps,
                    validation_steps=val_steps,
                    callbacks=[early_stopping])

loss, accuracy = model.evaluate(test_dataset, steps=test_steps)
print(f"Test Accuracy after training: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

predictions = model.predict(test_dataset, steps=test_steps)
y_pred = np.argmax(predictions.logits, axis=1)
y_true = test_data['fake'].values

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

print(f"Best Training Accuracy: {max(history.history['accuracy']) * 100:.2f}%")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")

model.save_pretrained("bert_fake_news_model")
tokenizer.save_pretrained("bert_fake_news_model")

print("Training complete and model saved.")
