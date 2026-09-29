import numpy as np 
from sklearn.datasets import fetch_california_housing ;

data = fetch_california_housing()
#Extratc the feature and the target from the data set
X = data.data
T = data.target.reshape(-1,1)

# Create the prediction vector F
theta = np.zeros((9,1))
one = np.ones((20640,1))
F = np.dot(np.hstack((X, one)), theta) # F shape: (m, 1) -> dot product between X_bias (m, n+1) and theta (n+1, 1)

#Function Lost 