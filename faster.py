import sys
import numpy as np
import pygame
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



ORIGIN_INDEX =  10
images = []
dades = []
with open('datathon/datathon/dataset/dades_processades.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('datathon/datathon/dataset/outfit_prep_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        outfit_data.append(row)

images = images[1:]
dades = dades[1:]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pygame.init()
width, height = 1200, 800  # Increase screen dimensions
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Outfit Visualizer")

# Set up fonts
font_large = pygame.font.Font(None, 48)  # Increase font size for main display
font_small = pygame.font.Font(None, 36)  # Font size for selection screen

# Function to display outfit images
def display_outfit(outfit_images):
    screen.fill((255, 255, 255))
    for i, image_path in enumerate(outfit_images):
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, ((width-100)/len(outfit_images), (width-100)/len(outfit_images)))  # Increase image scaling
        screen.blit(img, (i * (width-100)/len(outfit_images), 300))  # Adjust spacing and positioning

    pygame.display.flip()

def display_loading_screen():
    screen.fill((255, 255, 255))

    loading_text = font_large.render("Loading...", True, (0, 0, 0))
    screen.blit(loading_text, (width // 2 - loading_text.get_width() // 2, height // 2))

    pygame.display.flip()

# Function to display selection screen
def display_selection_screen(images):
    screen.fill((255, 255, 255))

    title_text = font_large.render("Choose Your First Clothing Piece", True, (0, 0, 0))
    screen.blit(title_text, (width // 2 - title_text.get_width() // 2, 100))

    instructions_text = font_small.render("Click on an item to select", True, (0, 0, 0))
    screen.blit(instructions_text, (width // 2 - instructions_text.get_width() // 2, 200))

    for i, image_path in enumerate(images):
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (150, 150))
        screen.blit(img, (i * 170, 300))

    pygame.display.flip()

# Get the initial outfit selection from the user
initial_images = images[500:530]  # Display the first 5 images on the selection screen
display_selection_screen(initial_images)

selected_item = None
while selected_item is None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            clicked_index = x // 170
            if 0 <= clicked_index < len(initial_images):
                selected_item = initial_images[clicked_index]
                display_loading_screen()
                pygame.display.flip()
                print("Loading...")
                break

print(selected_item)
ORIGIN_INDEX = 500+clicked_index

# Load your image embeddings
image_embeddings = np.load('image_embeddings_prep.npy')

outfit_data = outfit_data[1:]
metadata = dades
metadata = [m[:9] for m in metadata]
categorical_columns = [0,1,2,3,4,5,6,7,8]

# Extract the values from the categorical columns
categorical_values = [[entry[col] for col in categorical_columns] for entry in metadata]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(categorical_values)

metadata= onehot_encoded
# Function to calculate similarity between images based on metadata
def calculate_similarity_based_on_metadata(embedding1, embedding2, metadata1, metadata2):
    embedding_similarity = np.dot(embedding1, embedding2)
    metadata_similarity = cosine_similarity([metadata1], [metadata2])[0][0]

    combined_similarity = 0.5 * embedding_similarity + 0.5*metadata_similarity
    return combined_similarity

similarities = []
outfit = [ORIGIN_INDEX]

for e in range(len(image_embeddings)):
    metadata_similarity_score = calculate_similarity_based_on_metadata(
    image_embeddings[ORIGIN_INDEX],
    image_embeddings[e],
    metadata[ORIGIN_INDEX],
    metadata[e]
)
    similarities.append(metadata_similarity_score)



min_dist = np.argsort(similarities)


list_removed = []
Tipus_roba = set()

def outfit_complet(outfit):
    return len(Tipus_roba) == 6

def check_append(outfit, m, list_removed):
    accessories = 1
    if m in list_removed:
        return False
    i = True
    if dades[m][8] in Tipus_roba:
        return False
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
    return i

for o in outfit:
    if dades[o][8] == 'Accesories, Swim and Intimate':
        if dades[o][11] == 'Shoes':
            Tipus_roba.add('Shoes')
        elif 'Accesories, Swim and Intimate':
            Tipus_roba.add('Accesories, Swim and Intimate')
    else:
        Tipus_roba.add(dades[o][8])
    if dades[o][8] == 'Dresses, jumpsuits and Complete set':
        Tipus_roba.add('Tops')
        Tipus_roba.add('Bottoms')
    
    elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
        Tipus_roba.add('Dresses, jumpsuits and Complete set')

i = 1
while outfit_complet(outfit) == False and i < len(similarities):
    m = min_dist[-i]
    if check_append(outfit, m, list_removed):
        outfit.append(m)
        
        if dades[m][8] == 'Dresses, jumpsuits and Complete set':
            Tipus_roba.add('Tops')
            Tipus_roba.add('Bottoms')
        
        elif dades[m][8] == 'Tops' or dades[m][8] == 'Bottoms':
            Tipus_roba.add('Dresses, jumpsuits and Complete set')
        
        if dades[m][8] == 'Accesories, Swim and Intimate':
            print(dades[0][11])
            if dades[o][11] == 'Shoes':
                Tipus_roba.add('Shoes')
            elif 'Accesories, Swim and Intimate':
                Tipus_roba.add('Accesories, Swim and Intimate')
        else:
            Tipus_roba.add(dades[m][8])

    i += 1


desired_order = ['Dresses, jumpsuits and Complete set', 'Outerwear', 'Tops', 'Bottoms' ,'Accesories, Swim and Intimate']
outfit.sort(key=lambda x: desired_order.index(dades[x][8]))

outfit_images= [] 
for e in outfit:
     outfit_images.append( images[e])

outfit_complerts = []
for e in outfit:
    outfit_complerts.append( dades[e])


print(outfit_complerts)

gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

# Loop through the images and plot them
for i, image_path in enumerate(outfit_images):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

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
            clicked_index = x // ((width-100)//len(outfit_images))  # Adjust based on new spacing
            if 0 <= clicked_index < len(outfit):
                removed_item = outfit.pop(clicked_index)
                list_removed.append(removed_item)
                if dades[removed_item][8] == 'Accesories, Swim and Intimate' and  dades[removed_item][11] == 'Shoes':
                    Tipus_roba.remove('Shoes')
                else:
                    Tipus_roba.remove(dades[removed_item][8])
                print(f"Removed: {dades[removed_item][8]}")
        i = 1
        while outfit_complet(outfit) == False and i < len(similarities):
            m = min_dist[-i]
            if check_append(outfit, m, list_removed):
                outfit.append(m)
                if dades[m][8] == 'Accesories, Swim and Intimate':
                    print(dades[0][11])
                    if dades[o][11] == 'Shoes':
            
                        Tipus_roba.add('Shoes')
                    elif 'Accesories, Swim and Intimate':
                        Tipus_roba.add('Accesories, Swim and Intimate')
                else:
                    Tipus_roba.add(dades[m][8])
                if dades[o][8] == 'Dresses, jumpsuits and Complete set':
                    Tipus_roba.add('Tops')
                    Tipus_roba.add('Bottoms')
                
                elif dades[o][8] == 'Tops' or dades[o][8] == 'Bottoms':
                    Tipus_roba.add('Dresses, jumpsuits and Complete set')
            i += 1
        old_i = i
    outfit.sort(key=lambda x: desired_order.index(dades[x][8]))
    outfit_images = [images[e] for e in outfit]
    display_outfit(outfit_images)

    pygame.time.delay(30)