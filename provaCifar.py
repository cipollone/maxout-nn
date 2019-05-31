import numpy as np
import struct
import matplotlib.pyplot as plt



############################################ TUTTI I BATCH

#fileNames = ["datasets/cifar-10-batches-bin/data_batch_1.bin","datasets/cifar-10-batches-bin/data_batch_2.bin","datasets/cifar-10-batches-bin/data_batch_3.bin","datasets/cifar-10-batches-bin/data_batch_4.bin","datasets/cifar-10-batches-bin/data_batch_5.bin"]

test = ["datasets/cifar-10-batches-bin/test_batch.bin"]
fileNames = test
labelDs = []
imageDs = []
numeroFile =1 #temporaneo
all_img=np.empty((1,3072*10000*numeroFile))

for fileName in fileNames:
  with open(fileName) as my_file:

    allData = np.fromfile( my_file, dtype=np.uint8)
    lenFile= len(allData)

    print(lenFile)
    label = [allData[i] for i in range(lenFile) if i%3073==0]
    image = [allData[i] for i in range(lenFile) if i%3073!=0 and i!=0]
  imageDs.append(image)
  labelDs.append(label)
  #all_image= np.concatenate(image)  



#new_img = np.concatenate(imageDs)
new_img = np.reshape(imageDs,(10000*numeroFile,3072))
print(new_img.shape)
new_label = np.concatenate(labelDs)
print(new_label.shape)



with open ("newCifarlabelTEST.txt", "wb") as f:

  f.write(new_label)
  
with open ("newCifarImageTEST.txt", "wb") as f:

  f.write(new_img)
  

  
