import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


data = np.load('outputs/train_num.npy')
labels = np.load('outputs/train_labels.npy')

model = KNeighborsClassifier(n_neighbors=3)

KNN_Model = model.fit(data, labels)

with open('model.pkl', 'wb') as file:
    pickle.dump(KNN_Model, file)

print('Model training completed. Model has been saved')
