from datetime import datetime
from datetime import timedelta
from joblib import load, dump
from syncing import synced
from windowing import windowing_binary, windowing_multiclass
from features import features
from label_corrector import corrector
from os.path import dirname, join
import os
import pandas
import numpy
from dataframe_creation import dataframe_creation
from gravity import gravity_extraction
from labeling import add_labels
from classification import double_classifier

def to_time(i):
    seconds = round(i * 0.02)
    result = ""
    if seconds >= 60:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        if seconds != 0:
            result = str(minutes) + "' " + str(seconds) + "''"
        else:
            result = str(minutes) + "'"
    else:
        result = str(seconds) + "''"
    return result

def XGB():
    # Names of the files containing the raw data
    wristAcc = join(os.environ["HOME"]+"/wristAcc.csv")
    wristGyr = join(os.environ["HOME"]+"/wristGyr.csv")
    ankleAcc = join(os.environ["HOME"]+"/ankleAcc.csv")
    ankleGyr = join(os.environ["HOME"]+"/ankleGyr.csv")


    # Creating the dataframe with the features associated to the windows
    sync = synced(wristAcc, wristGyr, ankleAcc, ankleGyr)
    data = features(windowing_binary(sync))
    bool = load(join(dirname(__file__),"Mask.pkl"))
    data = data.loc[:,bool]

    # Classifying the windows
    xgb = load(join(dirname(__file__),"XGB.joblib"))
    labels = xgb.predict(data.values)

    # Correcting spottable mistakes
    labels = corrector(labels, 20)

    # # Creating output
    # output = ""
    # for i in range(len(labels) - 1):
    #     if labels[i] != labels[i + 1]:
    #         new = "From " + str(labels[i]) +  " to " + str(labels[i + 1]) + " at " + str(10 * i + 255)
    #         output = output + new
    # if output == "":
    #     output = "NOTHING"
    # return output

    # Creating the input for multiclass classifier
    rest = [True] * len(sync.index)
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            for j in range(10 * i + 255, len(sync.index)):
                if labels[i + 1] == "Rest" or labels[i + 1] == 0:
                    rest[j] = True
                else:
                    rest[j] = False
    exercise = [not elem for elem in rest]
    sync["Exercise"] = exercise

    # Multiclass classification
    df = dataframe_creation(sync)

    extracted_df = gravity_extraction(df)

    extracted_df = add_labels(extracted_df)

    training = extracted_df['Training']

    windowed_df = windowing_multiclass(extracted_df)

    predicted_labels = double_classifier(windowed_df)

    label = numpy.zeros(extracted_df.shape[0])
    i = 1
    j = 0

    while i < extracted_df.shape[0]:
        if training[i] == 1:
            label[i] = predicted_labels[j]
        else:
            if training[i-1] == 1:
                j = j+1
        i = i+1

    Label = []

    for i in range(len(label)):
        if label[i] == 0:
            Label.append('Rest')
        if label[i] == 1:
            Label.append('Push-Press')
        if label[i] == 2:
            Label.append('Shoulder-Press')
        if label[i] == 3:
            Label.append('Push-Jerk')
        if label[i] == 4:
            Label.append('Sit-Up')
        if label[i] == 5:
            Label.append('L-Up')
        if label[i] == 6:
            Label.append('V-Up')

    Label = [[element,i] for i, element in enumerate(Label) if element != Label[i-1]]
    for i, element in enumerate(Label):
        if element[0] == "Rest":
            Label[i-1].append(element[1]-1)
            Label.remove(element)

    result = ""
    for element in Label:
        element[1] = to_time(element[1])
        element[2] = to_time(element[2])
        result = result + ",".join(element) + "\n"
    return result
