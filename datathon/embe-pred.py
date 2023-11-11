import numpy as np
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
        images.append(str(e))

fclip = FashionCLIP('fashion-clip')
loaded_embeddings = np.load('image_embeddings.npy')

image_embeddings = fclip.encode_images(["C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/datathon/images/2019_53003780_OR.jpg"], batch_size=32)
ima = image_embeddings[0]
distances = []
for e in loaded_embeddings:
    distances.append(np.dot(ima, e))

outfit = ["C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/datathon/images/2019_53003780_OR.jpg"]
min_dist = np.argsort(distances)[:10]
for m in min_dist:
    outfit.append(images[m])

print(outfit)
gs = gridspec.GridSpec(3, 4, wspace=0.1, hspace=0.2)

# Loop through the images and plot them
for i, image_path in enumerate(outfit):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.set_title(f'Image {i + 1}')
    ax.axis('off')

plt.show()
