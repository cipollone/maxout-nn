import numpy as np
import struct
import matplotlib.pyplot as plt



############################################ TUTTI I BATCH

fileNames = ["datasets/my_cifar/before_shuffling/Image_TRAIN.txt","datasets/my_cifar/before_shuffling/Label_TRAIN.txt"]


#labelDs = []
#imageDs = []
#numeroFile =6 #temporaneo
#all_img=np.empty((1,3072*10000*numeroFile))


with open(fileNames[0]) as img:

  images = np.fromfile( img, dtype=np.uint8)
  
print(images.shape)

with open(fileNames[1]) as label:

  labels = np.fromfile(label , dtype=np.uint8)
print(labels.shape)
  
new_img= images.reshape((50000,3072))
new_label = labels.reshape((50000,1))

dataset= np.concatenate((new_img, new_label), axis=1)
#print(new_img.shape)
#print(new_label.shape)
#print(dataset.shape)

np.random.shuffle(dataset)
#print("#######after SHIFFLE######")
#print(new_img.shape)
#print(new_label.shape)
#print(dataset.shape)

train_data = dataset[0:45000,:]
#print(train_data.shape)
val_data = dataset[45000:,:] 
#print(val_data.shape)

##
train_img = train_data[:,:-1]
print(train_img.shape)
train_labels=train_data[:,-1]
print(train_labels.shape)
val_img = val_data[:,:-1]
print(val_img.shape)
val_labels = val_data[:,-1]
print(val_labels.shape)
##

with open("train_images.txt","wb") as f:
  train_img.tofile(f)

with open("train_labels.txt","wb") as f:
  train_labels.tofile(f)

with open("validation_images.txt","wb") as f:
  val_img.tofile(f)

with open("validation_labels.txt","wb") as f:
  val_labels.tofile(f)







