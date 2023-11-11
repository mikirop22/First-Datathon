
from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np

fclip = FashionCLIP('fashion-clip')
images = []
with open('datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        r = []
        for element in row:
            e = 'C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/' + element.replace('"', "")
            r.append(str(e))
        images.append(r)


print(images[0])
# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
print(image_embeddings[0])