import numpy as np
#from fashion_clip.fashion_clip import FashionCLIP
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

images = []
dades = []
with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/product_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuario/OneDrive/Escritorio/Datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)


images = images[1:]
dades = dades[1:]
#fclip = FashionCLIP('fashion-clip')
loaded_embeddings = np.load('image_embeddings.npy')

#image_embeddings = fclip.encode_images(["C:/Users/Usuari/OneDrive/Documentos/Datathon/aguacate/datathon/datathon/images/2019_53030677_32.jpg"], batch_size=32)
ima = loaded_embeddings[3256]
distances = []
for e in loaded_embeddings:
    distances.append(np.dot(ima, e))

#cloth = '"53090544-50"'
outfit = [3256]

#pos0 =  
Tipus_roba = []

def find_tipus(id):
    for a, l in enumerate(dades):
        if l[0] == id:
            return l[8]

min_dist = np.argsort(distances)

def outfit_complet(outfit):
    return len(Tipus_roba) == 5
    

def check_append(outfit, m):
    i = True
    if dades[m][8] in Tipus_roba:
        return False
    for o in outfit:
        if dades[o][8] == dades[m][8]:
            i = False
   
    return i

for o in outfit:
    Tipus_roba.append(dades[o][8])
    if dades[o][8] == '"Dresses':
        Tipus_roba.append('"Tops"')
        Tipus_roba.append('"Bottoms"')
    
    elif dades[o][8] == '"Tops"' or dades[o][8] == '"Bottoms"':
        Tipus_roba.append('"Dresses')
        
        
i = 1
while outfit_complet(outfit) == False:
    m = min_dist[-i]
    
    #print (dades[m][8],x)
    if check_append(outfit, m):
        outfit.append(m)
        Tipus_roba.append(dades[m][8])
        if dades[o][8] == '"Dresses':
            if '"Tops"' not in Tipus_roba and '"Bottoms"' not in Tipus_roba:    
                Tipus_roba.append('"Tops"', '"Bottoms"')
        
        elif dades[o][8] == '"Tops"' or dades[o][8] == '"Bottoms"':
            if '"Dresses' not in Tipus_roba:   
                Tipus_roba.append('"Dresses')

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

