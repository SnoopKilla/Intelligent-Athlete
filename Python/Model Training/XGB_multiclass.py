import pandas
import numpy
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
from xgboost import XGBClassifier
import pickle
import joblib

X = joblib.load('C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Data/TrainingSet_Sho.pkl')
# X = joblib.load('C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Data/TrainingSet_Abs.pkl')
athletes = X['Athlete']
athletes = athletes.astype('category')
names = athletes.unique()
col_names = X.iloc[:,:-3].columns

# Using only selected features
filename = open('C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Python/Classifier/Models/Multiclass/SHOmask25', 'rb')
# filename = open('C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Python/Classifier/Models/Multiclass/ABSmask25', 'rb')
mask = numpy.load(filename)
filename.close()

col_names_new = col_names[mask]
col_names_new = list(col_names_new)
col_names_new.append('Label')
col_names_new.append('Athlete')
col_names_new.append('Session')

X_new = X[col_names_new]

# Training of the classifier
y = X_new['Label']
X_new.drop('Label', inplace = True, axis = 1)
X_new.drop('Athlete', inplace = True, axis = 1)
X_new.drop('Session', inplace = True, axis = 1)

clf_final = XGBClassifier()
clf_final.fit(X_new, y)

# Save the classifier
with open('classifier_sho.pkl', 'wb') as fid:
    pickle.dump(clf_final, fid)
# with open('classifier_abs.pkl', 'wb') as fid:
#     pickle.dump(clf_final, fid)
