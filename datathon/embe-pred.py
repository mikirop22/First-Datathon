import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

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
print(images[0])
dades = dades[1:]
loaded_embeddings = np.load('image_embeddings.npy')

ima = loaded_embeddings[3]
distances = []
for e in loaded_embeddings:
    distances.append(np.dot(ima, e))

outfit = [3]


min_dist = np.argsort(distances)

def outfit_complet(outfit):
    return len(outfit) == 8

def check_append(outfit, m):
    accessories = 1
    i = True
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            if dades[m][8] != '"Accesories' or accessories == 2:
                i = False
            else:
                accessories += 1
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

print(outfit)
gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

# Loop through the images and plot them
for i, image_path in enumerate(outfit_images):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

plt.show()
