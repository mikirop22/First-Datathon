import csv
images = []
dades = []
with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/dades_processades.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        element = row[-1]
        e = 'C:/Users/Usuario/OneDrive/Escritorio/Datathon/' + element.replace('"', "")
        images.append(str(e))
        dades.append(row)

outfit_data = []
with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/outfit_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[1] in [d[0] for d in dades]:
            outfit_data.append(row)

with open('C:/Users/Usuario/OneDrive/Escritorio/Datathon/datathon/dataset/outfit_prep_data.csv', 'w', newline='') as f:
    write = csv.writer(f,  )
    for row in outfit_data:
        write.writerow(row)
