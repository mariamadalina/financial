import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from scipy import signal
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import MinMaxScaler
from datetime import tzinfo, timedelta, datetime


from Pipeline import CustomPypeline
from features.sources import FileDataSource, WebDataSource


start_date = datetime(2010, 8, 3)
end_date = datetime(2015, 5, 23)


dataSource = FileDataSource('data/ES 1 Day_Series_Indicators.csv')
df = dataSource.Data()
#df=ds.WebDataSource(start_date,end_date,'FB').Data()
info = dataSource.GetMissingValueInfo(df)
print(info)

# # load dataset
# X, y = CustomPypeline.load_dataset()
# # get model list
# models = define_models()

# # add gbm models
# models = define_gbm_models(models)

# # evaluate models
# results = evaluate_models(X, y, models)
# # summarize results
# summarize_results(results)

X=df

numberOfFeatures=5
# normalize the data 
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
print('Min: {0}, Max: {1}'.format(scaler.data_min_, scaler.data_max_))
# normalize the dataset and print the first 5 rows
X = scaler.transform(X)

ica = FastICA(n_components=numberOfFeatures,fun='logcosh',whiten=True)
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
    
