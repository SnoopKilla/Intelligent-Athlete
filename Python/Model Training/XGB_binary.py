import pandas
import numpy
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from joblib import load, dump
from sklearn.feature_selection import RFE
from statistics import mean

def label_encoder(label):
    return 1 if label == "Exercise" else 0

def future_check(labels, i, n):
    result = True
    for j in range(n):
        try:
            to_check = labels[i + j + 1]
        except:
            break
        if labels[i] != to_check:
            result = False
    return result

def corrector(labels, n):
    current = labels[0]
    for i in range(len(labels)):
        if labels[i] != current:
            if future_check(labels, i, n):
                current = labels[i]
            else:
                labels[i] = current
    return labels

# XGB Classifier
data = pandas.read_pickle('C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Data/TrainingSet_Binary.pkl')

# Features and Labels
X = data[data.columns[:-3]]
y = data[data.columns[-3]]
y = pandas.DataFrame([label_encoder(i) for i in y])
y = y.values.ravel()

# Selected Features
bool = load("C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Python/Classifier/Models/Binary/Mask.pkl")
X = X.loc[:, bool]
X = X.values

# Leave-One-Subject-Out CV
n = 0
accuracy = 0
for athlete in set(data["Athlete"]):
    selector = (data["Athlete"] == athlete)
    X_train = X[~selector]
    X_test = X[selector]
    y_train = y[~selector]
    y_test = y[selector]
    clf = XGBClassifier()
    clf = clf.fit(X_train, y_train)
    accuracy += clf.score(X_test, y_test)
    n += 1
print("Leave-One-Subject-Out CV: ", accuracy / n)

# # Save the classifier
# clf = XGBClassifier()
# clf = clf.fit(X, y)
# dump(clf, "C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Python/Classifier/Models/Binary/XGB.joblib")

# # Feature Selection
# clf = XGBClassifier()
# selector = RFE(clf, n_features_to_select = 45, step=1)
# selector = selector.fit(X, y)
# print(selector.support_)
# dump(selector.support_,"C:/Users/russo/OneDrive/Documents/GitHub/Intelligent-Athlete/Python/Classifier/Models/Binary/Mask.pkl")

# # Delay analysis
# shifts = list()
# for athlete in set(data["Athlete"]):
#     df = data[data["Athlete"] == athlete]
#     for session in set(df["Session"]):
#         print(athlete, session)
#         selector = (data["Athlete"] == athlete) & (data["Session"] == session)
#         X_train = X[~selector]
#         X_test = X[selector]
#         y_train = y[~selector]
#         y_test = y[selector]
#         clf = XGBClassifier()
#         clf = clf.fit(X_train, y_train)
#         print("Accuracy: ", clf.score(X_test,y_test))
#         y_label = clf.predict(X_test)
#         y_label = corrector(y_label, 1)
#         shift = 0
#         sign = 0
#         y_test = y_test.tolist()
#         temp = list()
#         for index, (true, predicted) in enumerate(zip(y_test, y_label)):
#             if true != predicted:
#                 if shift == 0:
#                     if y_test[index] != y_test[index-1]:
#                         sign = +1
#                     else:
#                         sign = -1
#                 shift += 1
#             elif true == predicted and shift != 0:
#                 print(sign * shift)
#                 temp.append(sign * shift)
#                 shift = 0
#         other = [0] * (6 - len(temp))
#         shifts.append(temp)
#         shifts.append(other)
# shifts = [item for sublist in shifts for item in sublist]
# plt.hist(shifts, bins = numpy.arange(-4.5,5,1))
# plt.axvline(x = mean(shifts), color='k', linestyle='dashed', linewidth=1)
# plt.xlabel("Delay (Number of Windows)")
# plt.ylabel("Occurencies")
# plt.show()
# print(mean(shifts))
