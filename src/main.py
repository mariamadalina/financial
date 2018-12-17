import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import tzinfo, timedelta, datetime

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from features.sources import FileDataSource

start_date = datetime(2010, 8, 3)
end_date = datetime(2015, 5, 23)
numberOfFeatures = 5
colors = ['red', 'steelblue', 'orange', 'teal', 'green']


dataSource = FileDataSource('data/ES 1 Day_Series_Indicators.csv')
df = dataSource.dataFrame
train_data, train_data_target = dataSource.GetData(), dataSource.Target

pipeline = Pipeline([
    ('normalize', MinMaxScaler(feature_range=(-1, 1))),
    ('featureExtraction', FastICA(
        n_components=numberOfFeatures, fun='logcosh', whiten=True))
])

pipeline = pipeline.fit(train_data)
S = pipeline.transform(train_data)


# # evaluate models
# results = CustomPipeline.evaluate_models(X, y, models, pipelines)
# # summarize results
# CustomPipeline.summarize_results(results)

X = df.values
# normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
# normalize the dataset and print the first 5 rows
X = scaler.transform(X)

ica = FastICA(n_components=numberOfFeatures, fun='logcosh', whiten=True)
S_ = ica.fit_transform(X)  # Reconstruct signals

# #############################################################################
# Plot results

df.plot()


plt.figure()

signals = {
    'Observations (mixed signal)': X,
    'ICA recovered signals': S_,
    'Pipeline fit signal': S
}
colors = ['red', 'steelblue', 'orange', 'teal', 'green']

for idx, key in enumerate(signals, 1):

    plt.subplot(len(signals), 1, idx)
    plt.title(key)
    for sig, color in zip(signals[key].T, colors):
        plt.plot(sig, color=color)

plt.show()
