import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# scaled test values - see process.py for details
X = np.load('outputs/test_num.npy')
y_test = np.load('outputs/test_labels.npy')

y_hat = model.predict(X)

print('f1:', f1_score(y_test, y_hat, average='micro'))
print('precision:', precision_score(y_test, y_hat, average='macro'))
print('recall:', recall_score(y_test, y_hat, average='macro'))
print('')
print(classification_report(y_test, y_hat))
print(confusion_matrix(y_test, y_hat))
