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

fclip = FashionCLIP('fashion-clip')
loaded_embeddings = np.load('image_embeddings.npy')

image_embeddings = fclip.encode_images(["C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/datathon/images/2019_53030677_32.jpg"], batch_size=32)
ima = image_embeddings[0]
distances = []
for e in loaded_embeddings:
    distances.append(np.dot(ima, e))

cloth = "53090544-50"
outfit = [1]

def find_tipus(id):
    for a, l in enumerate(dades):
        if l[0] == id:
            return l[8]
tipus = find_tipus(cloth)

min_dist = np.argsort(distances)

def outfit_complet(outfit):
    return len(outfit) == 6


def check_append(outfit, m):
    i = True
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
    return i


i = 0
while outfit_complet(outfit) == False:
    m = min_dist[i]
    if check_append(outfit, m):
        outfit.append(m)
    i += 1

for e in outfit:
    print(dades[e][8])

outfit_images= [] 
for e in outfit:
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
