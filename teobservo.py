import pygame
import sys
import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import csv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


ORIGIN_INDEX =  40
images = []
dades = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        outfit_data.append(row)

images = images[1:]
dades = dades[1:]
loaded_embeddings = np.load('image_embeddings.npy')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your image embeddings
image_embeddings = np.load('image_embeddings.npy')

outfit_data = outfit_data[1:]
metadata = dades
combined = []

for ima in dades:
    aa = []
    for ou in outfit_data:
        if ou[1] == ima[0]:
            aa.append(ou[0])
    combined.append(aa)
meta_outfits = combined

#Outfit mean embedings:
outfit_embeddings = {}
codis = [m[0] for m in metadata]
# Calculate the average embedding for each outfit
for outfit in outfit_data:
    outfit_id, item_id = outfit
    item_idx = codis.index(item_id)
    item_embedding = image_embeddings[item_idx]

    if outfit_id not in outfit_embeddings:
        outfit_embeddings[outfit_id] = item_embedding
    else:
        outfit_embeddings[outfit_id] += item_embedding

# Normalize outfit embeddings
for outfit_id, embedding in outfit_embeddings.items():
    outfit_embeddings[outfit_id] /= len(embedding)



metadata = [m[:9] for m in metadata]
categorical_columns = [0,1,2,3,4,5,6,7,8]

# Extract the values from the categorical columns
categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categorical_values)

metadata= onehot_encoded
print(metadata[0])
# Function to calculate similarity between images based on metadata
def calculate_similarity_based_on_metadata(embedding1, embedding2, metadata1, metadata2, metaoutfits1, metaoutfits2):
    # For simplicity, use cosine similarity
    embedding_similarity = np.dot(embedding1, embedding2)
    # Calculate cosine similarity between metadata
    metadata_similarity = cosine_similarity([metadata1], [metadata2])[0][0]
    
    meta1 = [None]
    for m1 in metaoutfits1:
        if meta1[0] == None:
            meta1 = outfit_embeddings[m1]
        else:
            meta1 += outfit_embeddings[m1]
    meta1 /= len(metaoutfits1)
    meta2 = [None]
    for m2 in metaoutfits2:
        if meta2[0]== None:
            meta2 = outfit_embeddings[m2]
        else:
            meta2 += outfit_embeddings[m2]
    meta2 /= len(metaoutfits2)
    outfits_similarity = np.dot(meta1, meta2)
    
    # Combine similarities (you can adjust the weights based on importance)
    combined_similarity = 0.5 * embedding_similarity + 0.1* metadata_similarity + 0.4* outfits_similarity if outfits_similarity != float("inf") else 0.6 * embedding_similarity + 0.4* metadata_similarity
    return combined_similarity

# Example: Calculate similarity between the first two images based on metadata



similarities = []

for e in range(len(loaded_embeddings)):
    metadata_similarity_score = calculate_similarity_based_on_metadata(
    image_embeddings[ORIGIN_INDEX],
    image_embeddings[e],
    metadata[ORIGIN_INDEX],
    metadata[e],
    meta_outfits[ORIGIN_INDEX],
    meta_outfits[e]
)
    similarities.append(metadata_similarity_score)


outfit = [ORIGIN_INDEX]
min_dist = np.argsort(similarities)



def outfit_complet(outfit):
    return len(outfit) == 8

list_removed = []
def check_append(outfit, m, list_removed):
    accessories = 1
    if m in list_removed:
        return False
    i = True
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
    return i


i = 1
while outfit_complet(outfit) == False and i < len(similarities):
    m = min_dist[-i]
    if check_append(outfit, m, list_removed):
        outfit.append(m)
    i += 1

for e in outfit:
    print(dades[e][8])

outfit_images= [] 
for e in outfit:
     outfit_images.append( images[e])

outfit_complerts = []
for e in outfit:
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


pygame.init()

# Set up screen
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Outfit Visualizer")

# Set up fonts
font = pygame.font.Font(None, 36)

# Function to display outfit images
def display_outfit(outfit_images):
    screen.fill((255, 255, 255))
    for i, image_path in enumerate(outfit_images):
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (100, 100))
        screen.blit(img, (i * 120, 200))

    pygame.display.flip()

old_i = 1
min_dist = np.argsort(similarities)
# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            clicked_index = x // 120  # Assuming each outfit image has a width of 120 pixels
            if 0 <= clicked_index < len(outfit):
                removed_item = outfit.pop(clicked_index)
                list_removed.append(removed_item)
                print(f"Removed: {dades[removed_item][8]}")
    i = 0
    while outfit_complet(outfit) == False and i < len(similarities):
        m = min_dist[-i]
        if check_append(outfit, m, list_removed):
            outfit.append(m)
        print("a")
        i += 1
    old_i = i
    print(old_i)
    print(outfit)
    outfit_images = [images[e] for e in outfit]
    display_outfit(outfit_images)

    pygame.time.delay(100)  # Add a delay to control the refresh rate
