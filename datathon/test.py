import csv
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

combined = []
maxlen = 0
for ima in images[1:2000]:
    for ou in outfits[1:200]:
        if ou[1] == ima[0]:
            seq = ["START"]
            for ou2 in outfits[1:200]:
                if ou[0] == ou2[0]:
                    seq.append(ou2[-1])
            seq.append("END")
            if len(seq) > maxlen:
                maxlen = len(seq)
            combined.append([ima[-1], seq])
print(combined)
with open('datathon/datathon/dataset/combined_data.csv', 'w', newline='') as f:
    write = csv.writer(f,  )
    for row in combined:
        write.writerow(row)