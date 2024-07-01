import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from Bio import SeqIO

# Load the dataset
def load_data(file_path):
    sequences = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        labels.append(int(record.id.split('|')[1]))  # assuming label is in the ID
    return pd.DataFrame({'sequence': sequences, 'label': labels})

data = load_data('ion_channel_sequences.fasta')

# Preprocess data: Create word embeddings
def preprocess_sequences(sequences, vocab_size=20, max_length=100):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return sequences, tokenizer

sequences, tokenizer = preprocess_sequences(data['sequence'])
labels = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Build the 5-layer neural network
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=100))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Test Accuracy: {accuracy:.2f}')
print(f'F1 Score: {f1:.2f}')

# Plot the training history
history = model.history.history
plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
