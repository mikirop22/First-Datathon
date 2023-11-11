import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import cv2
import csv

# Load data
images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        images.append(row)

outfits = []
with open('datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfits.append(row)

# Combine data
combined = []
maxlen = 0
for ima in images:
    for ou in outfits:
        if ou[1] == ima[0]:
            seq = ["START"]
            for ou2 in outfits:
                if ou[0] == ou2[0]:
                    seq.append(ou2[-1])
            if len(seq) > maxlen:
                maxlen = len(seq)
            combined.append([ima[-1], seq])

# Save combined data to CSV
with open('datathon/datathon/dataset/combined_data.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(combined)

# Tokenize and pad sequences
sequence_data = [[e.replace('"', "") for e in a] for _, a in combined]

z = []
for e in sequence_data:
    for l in e:
        z.append(l)
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([str(seq) for seq in z])
num_classes = len(tokenizer.word_index) + 1

X_images = []
X_seq = []
y_seq = []

# Prepare data for training
for yes in range(len(combined)):
    sequ = sequence_data[yes]
    for s in range(1, len(sequ)-1):
        X_images.append(cv2.imread('datathon/' + combined[yes][0].replace('"', "")))
        X_seq.append(sequ[:s])
        y_seq.append(str(sequ[s+1]).replace('"', ''))

# Tokenize and pad input sequences
tokenized_sequences = tokenizer.texts_to_sequences([" ".join(seq) for seq in X_seq])
x_padded = pad_sequences(tokenized_sequences, maxlen=num_classes, padding='post', truncating='post')

# Tokenize and pad output sequences
tokenized_sequences_y = tokenizer.texts_to_sequences(y_seq)
# One-hot encode the tokenized output sequences
y_one_hot = to_categorical(tokenized_sequences_y, num_classes=num_classes)


# Define the image branch
image_input = keras.Input(shape=(334, 239, 3))  # Adjust input shape based on your images
encoded_image = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
encoded_image = layers.Flatten()(encoded_image)

# Define the sequence branch
sequence_input = keras.Input(shape=(num_classes,))
embedding_dim = 50
embedded_sequence = layers.Embedding(num_classes, embedding_dim, input_length=maxlen)(sequence_input)
lstm_output = layers.LSTM(256)(embedded_sequence)

# Combine the branches
merged = layers.concatenate([encoded_image, lstm_output])

# Add more dense layers for final prediction
output = layers.Dense(num_classes, activation='softmax')(merged)

# Create the model
model = keras.Model(inputs=[image_input, sequence_input], outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x=[np.asarray(X_images), x_padded],
    y=y_one_hot,
    epochs=100,
    batch_size=1024,
    validation_split=0.2
)

# Function to generate the next element of the sequence
def generate_next_element(model, current_image, current_sequence, tokenizer, maxlen):
    # Preprocess the current image
    current_image = cv2.imread(current_image.replace('"', ""))

    # Tokenize and pad the current sequence
    current_sequence = [current_sequence]
    tokenized_sequence = tokenizer.texts_to_sequences([" ".join(seq) for seq in current_sequence])
    x_padded = pad_sequences(tokenized_sequence, maxlen=num_classes, padding='post', truncating='post')

    # Predict the next element
    predictions = model.predict([np.expand_dims(current_image, axis=0), x_padded])
    predicted_index = np.argmax(predictions)
    
    # Reverse lookup to get the predicted word
    predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_index][0]

    return predicted_word

# Initialize with a starting image and an initial sequence
current_image_path = 'datathon/' + combined[0][0].replace('"', "")
current_sequence = ["START"]

# Generate the next 5 elements of the sequence
for _ in range(5):
    next_element = generate_next_element(model, current_image_path, current_sequence, tokenizer, maxlen)
    current_sequence.append(next_element)

# Display the generated sequence
print("Generated Sequence:", current_sequence)

# Save the model
model.save('./model.keras')
