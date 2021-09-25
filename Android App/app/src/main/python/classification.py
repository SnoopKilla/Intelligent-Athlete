import numpy
import joblib
from scipy.stats import mode
from os.path import dirname, join

def double_classifier(windowed_df):

    colnames = windowed_df.iloc[:,:-1].columns

    # Carico le 2 maschere per spalle e addominali e i rispettivi classificatori salvati
    filename1 = open(join(dirname(__file__),'SHOmask25'), 'rb')
    mask_sho = numpy.load(filename1)
    filename1.close()
    colnames_sho = colnames[mask_sho]
    colnames_sho = list(colnames_sho)

    filename2 = open(join(dirname(__file__),'ABSmask25'), 'rb')
    mask_abs = numpy.load(filename2)
    filename2.close()
    colnames_abs = colnames[mask_abs]
    colnames_abs = list(colnames_abs)

    clf_sho = joblib.load(join(dirname(__file__),'classifier_sho.pkl'))
    clf_abs = joblib.load(join(dirname(__file__),'classifier_abs.pkl'))

    # Faccio predizione su due livelli per ogni singolo esercizio
    exercises = windowed_df['Exercise']
    predicted_labels = []

    for ex in range(int(numpy.max(exercises))):

        X_test = windowed_df[exercises == ex + 1]
        X_test.drop('Exercise', inplace = True, axis = 1)

        # primo livello di classificazione: spalle o addominali
        shoabs = X_test['m5'] > -0.8
        pred_shoabs = mode(shoabs)[0][0]

        # a seconda dell'esito chiamo il classificatore corretto
        if pred_shoabs == 0:  # esercizio di spalle

            y_pred = clf_sho.predict(X_test[colnames_sho])
            most_frequent = mode(y_pred)[0][0]
            predicted_labels.append(most_frequent)

        else:  # esercizio di addominali

            y_pred = clf_abs.predict(X_test[colnames_abs])
            most_frequent = mode(y_pred)[0][0]
            predicted_labels.append(most_frequent)

    return predicted_labels
