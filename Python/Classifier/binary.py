from joblib import load
from syncing import synced
from windowing import windowing_binary
from label_corrector import corrector

def classifier(wA,wG,aA,aG):
    # Creating the dataframe with the features associated to the windows
    dataSynced, start = synced(wA, wG, aA, aG)
    dataWindowed = windowing_binary(dataSynced)
    bool = load("Models/Binary/Mask.pkl")
    dataWindowed = dataWindowed.loc[:, bool]

    # Classifying the windows
    xgb = load("Models/Binary/XGB.joblib")
    labels = xgb.predict(dataWindowed.values)

    # Correcting spottable mistakes
    labels = corrector(labels, 10)

    # Creating output
    rest = [True] * len(dataSynced.index)
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            # print("From ", labels[i], " to ", labels[i + 1], " at ", 10 * i + 255)
            for j in range(10 * i + 255, len(dataSynced.index)):
                if labels[i + 1] == "Rest" or labels[i + 1] == 0:
                    rest[j] = True
                else:
                    rest[j] = False
    exercise = [not elem for elem in rest]
    dataSynced["Exercise"] = exercise

    return dataSynced
