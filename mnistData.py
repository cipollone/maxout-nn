import numpy as np
import struct

def read_mnist(images_name,labels_name):

  with open(labels_name, 'rb') as lab:
    magic, n = struct.unpack('>II',lab.read(8))
    labels = np.fromfile(lab, dtype=np.uint8)


  with open(images_name, "rb") as img:
    magic, num, rows, cols = struct.unpack(">IIII",img.read(16))
    images = np.fromfile( img, dtype=np.uint8).reshape(len(labels), 784)

  return images,labels


    #print(ds_labels)


## image plot example
#my_img = np.resize(my_images[7],(28,28))
#plt.imshow(my_img)
#plt.show()

