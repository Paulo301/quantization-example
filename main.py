from skimage import data, io
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

n_colors = 2

astronautImg = data.astronaut()

#image_raw = io.imread("imagename") para fazer os valores de uma imagem do computador ficarem entre 0 e 255
image_raw = astronautImg
image = np.array(image_raw, dtype=np.float64)/255 
#Usado para pegar a forma da array
h, w, d = image.shape
#Muda a forma da array sem mudar os valores
image_array = np.reshape(image, (h*w, d))

#Imagem original
plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Imagem Original")
plt.imshow(image)

#Histograma Imagem Original - Valores na escala cinza
plt.figure(2)
plt.clf()
plt.title("Histograma da Imagem Original")
hist = np.histogram(image_raw, bins=np.arange(0,256))
plt.plot(hist[1][:-1], hist[0], lw=2)

image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors).fit(image_array_sample)
labels = kmeans.predict(image_array)

image_out = np.zeros((h, w, d))
label_idx = 0
#Percorrendo linhas e colunas
for i in range(h):
  for j in range(w):
    image_out[i][j] = kmeans.cluster_centers_[labels[label_idx]]
    label_idx += 1

#Imagem Quantizada
plt.figure(3)
plt.clf()
plt.axis("off")
plt.title("Imagem Original")
plt.imshow(image_out)

image_out = np.array(image_out * 255, dtype=np.uint8)

#Histograma Imagem Quantizada - Valores na escala cinza
plt.figure(4)
plt.clf()
plt.title("Histograma da Imagem Quantizada")
hist = np.histogram(image_out, bins=np.arange(0,256))
plt.plot(hist[1][:-1], hist[0], lw=2)

plt.show()