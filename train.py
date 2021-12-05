from os import pipe
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

dir_path = os.path.dirname(os.path.realpath(__file__))

data = np.load(os.path.join(dir_path, 'outputs/train_num.npy'))
labels = np.load(os.path.join(dir_path, 'outputs/train_labels.npy'))

model = KNeighborsClassifier(n_neighbors=3)
pipeline = joblib.load(os.path.join(dir_path, 'pipe_fit.joblib'))

pipeline.steps.append(['train', model])
pipeline.fit(data, labels)

joblib.dump(pipeline, os.path.join(dir_path, 'knn_model.joblib'))

print('Model training completed. Model has been saved')
