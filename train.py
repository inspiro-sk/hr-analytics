from os import pipe
import pickle
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


data = np.load('outputs/train_num.npy')
labels = np.load('outputs/train_labels.npy')

model = KNeighborsClassifier(n_neighbors=3)
pipeline = joblib.load('pipe_fit.joblib')

pipeline.steps.append(['train', model])
pipeline.fit(data, labels)

joblib.dump(pipeline, 'knn_model.joblib')

print('Model training completed. Model has been saved')
