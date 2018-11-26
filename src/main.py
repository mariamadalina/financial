import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from scipy import signal
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import MinMaxScaler
from datetime import tzinfo, timedelta, datetime
import FileDataSource as fileDataSource
import WebDataSource as dataSource

start_date = datetime(2010, 8, 3)
end_date = datetime(2015, 5, 23)

df = dataSource.WebDataSource('FB',start_date,end_date).Data
# other=fileDataSource.FileDataSource('data/ES 1 Day_Series_Indicators.csv',start_date,end_date).Data

numberOfFeatures=5
X = df.iloc[:,0:numberOfFeatures].values
print("Value of X \n {0}".format(X))
print(X.shape)
print("Standard deviation:  {0} with shape{1}".format(X.std(axis=0),X.std(axis=0).shape))

#X = np.multiply(X,1/X.std(axis=0))


# normalize the data 
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
print('Min: {0}, Max: {1}'.format(scaler.data_min_, scaler.data_max_))
# normalize the dataset and print the first 5 rows
X = scaler.transform(X)

ica = FastICA(n_components=numberOfFeatures)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix


# For comparison, compute PCA
pca = PCA(n_components=numberOfFeatures)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

print(X.shape,S_.shape,H.shape)
print(X[0:3,:])
print(S_[0:3,:])
print(H[0:3,:])

df.plot()



models = [X, S_, H]
plt.figure()
names = ['Observations (mixed signal)',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange','teal','green']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.show()
    
