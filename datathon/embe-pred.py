import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



images = []
dades = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

images = images[1:]
print(images[4])
dades = dades[1:]
loaded_embeddings = np.load('image_embeddings.npy')

ima = loaded_embeddings[123]
distances = []
for e in loaded_embeddings:
    distances.append(np.dot(ima, e))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your image embeddings
image_embeddings = np.load('image_embeddings.npy')
metadata = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        metadata.append(row)

metadata = metadata[1:]

metadata = [m[2:11] for m in metadata]
categorical_columns = [0,1,2,3,4,5,6,7,8]

# Extract the values from the categorical columns
categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]

# One-hot encode categorical values
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categorical_values)

metadata= onehot_encoded
print(metadata[0])
# Function to calculate similarity between images based on metadata
def calculate_similarity_based_on_metadata(embedding1, embedding2, metadata1, metadata2):
    # For simplicity, use cosine similarity
    embedding_similarity = np.dot(embedding1, embedding2)
    
    # Calculate cosine similarity between metadata
    metadata_similarity = cosine_similarity([metadata1], [metadata2])[0][0]
    
    # Combine similarities (you can adjust the weights based on importance)
    combined_similarity = 0.7 * embedding_similarity + 0.3 * metadata_similarity
    return combined_similarity

# Example: Calculate similarity between the first two images based on metadata



similarities = []

for e in range(len(loaded_embeddings)):
    metadata_similarity_score = calculate_similarity_based_on_metadata(
    image_embeddings[123],
    image_embeddings[e],
    metadata[123],
    metadata[e]
)
    similarities.append(metadata_similarity_score)


outfit = [123]


min_dist = np.argsort(similarities)
print(min_dist)

def outfit_complet(outfit):
    return len(outfit) == 8

def check_append(outfit, m):
    accessories = 1
    i = True
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
    return i


i = 1
while outfit_complet(outfit) == False and i < len(distances):
    m = min_dist[-i]
    if check_append(outfit, m):
        outfit.append(m)
    i += 1

for e in outfit:
    print(dades[e][8])

outfit_images= [] 
for e in outfit:
    if e != 0:
        outfit_images.append( images[e])

outfit_complerts = []
for e in outfit:
    if e != 0:
        outfit_complerts.append( dades[e])

print(outfit)
gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

print(outfit_images)
print(outfit_complerts)
# Loop through the images and plot them
for i, image_path in enumerate(outfit_images):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

plt.show()
