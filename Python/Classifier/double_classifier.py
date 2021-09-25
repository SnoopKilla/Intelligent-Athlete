import numpy
import joblib
from scipy.stats import mode

def classifier(windowed_df):
    colnames = windowed_df.iloc[:,:-1].columns

    # Load mask of selected features and classifiers for abs and shoulders
    filename1 = open('Models/Multiclass/SHOmask25', 'rb')
    mask_sho = numpy.load(filename1)
    filename1.close()
    colnames_sho = colnames[mask_sho]
    colnames_sho = list(colnames_sho)
    clf_sho = joblib.load('Models/Multiclass/classifier_sho.pkl')

    filename2 = open('Models/Multiclass/ABSmask25', 'rb')
    mask_abs = numpy.load(filename2)
    filename2.close()
    colnames_abs = colnames[mask_abs]
    colnames_abs = list(colnames_abs)
    clf_abs = joblib.load('Models/Multiclass/classifier_abs.pkl')

    # Classification of the session on two levels
    exercises = windowed_df['Exercise']
    predicted_labels = []

    for ex in range(int(numpy.max(exercises))):
        X_test = windowed_df[exercises == ex + 1]
        X_test.drop('Exercise', inplace = True, axis = 1)

        # First level of classification (shoulder vs abs)
        shoabs = X_test['m5'] > -0.8
        pred_shoabs = mode(shoabs)[0][0]

        # Second level of classification
        if pred_shoabs == 0:  # shoulder
            y_pred = clf_sho.predict(X_test[colnames_sho])
            most_frequent = mode(y_pred)[0][0]
            predicted_labels.append(most_frequent)

        else:  # abs
            y_pred = clf_abs.predict(X_test[colnames_abs])
            most_frequent = mode(y_pred)[0][0]
            predicted_labels.append(most_frequent)

    return predicted_labels
