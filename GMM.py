from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt
import cv2


n=int(input('Enter number of clusters\n'))
print('Enter\n1 to select k-means clustering\n2 to select GMM')
cmd=int(input('enter the desired image segmentation algorithm\n'))

img = cv2.imread("road.png")
plt.imshow(img)
# For clustering, 2D array is required. -1 reshape means, in this case MxN
img_2D = img.reshape((-1, 3))  #For clustering, 3D array is converted into 2D array


if cmd == 1:
    
    print('Initiating K-means clustering')
    
    kmeans = KMeans(n, init='k-means++', max_iter=250, n_init=10, random_state=35)
    # k-means++ avoids random initialization trap
    model = kmeans.fit(img_2D) #as per img2 shape
    values = kmeans.predict(img_2D)

    mask1 = values.reshape((img.shape[0], img.shape[1])) # 2D array
    plt.imshow(mask1) # masked cluster of k means
    mask1 = np.expand_dims(mask1, axis=-1) # to make 2D to 3D array
    plt.show()
    
    foreground = np.multiply(mask1, img)
    background = img - foreground
    
    plt.imshow(foreground) 
    plt.show()
    plt.imshow(background)
    plt.show()
      
elif cmd == 2:
    
    print('Initiating GMM')
    
    gmm_model = GMM(n, covariance_type='tied').fit(img_2D)  
    gmm_labels = gmm_model.predict(img_2D)
    
    #Put numbers back to original shape so we can reconstruct segmented image
    original_shape = img.shape
    mask2 = gmm_labels.reshape(original_shape[0], original_shape[1]) #2D array
    mask2 = np.expand_dims(mask2, axis=-1)  ## to make 2D to 3D array
    plt.imshow(mask2) #masked cluster of GMM
    plt.show()
    foreground = np.multiply(mask2, img)
    background = img - foreground
    
    plt.imshow(foreground) 
    plt.show()
    plt.imshow(background)
    plt.show()
 

else:
    print('Invalid input')
    


