import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(10, 10), dpi=90)
n=1
while n<500:
    pca= PCA(n_components=n)
    array=np.asarray(Image.open('cosmo1.jpg'))
    pca_reduced = pca.fit_transform(array)
    pca_recovered = pca.inverse_transform(pca_reduced)
    array=pca_recovered
    plt.imshow(array, cmap='gray_r')
    plt.savefig(str(n)+'.png')
    n+=1
