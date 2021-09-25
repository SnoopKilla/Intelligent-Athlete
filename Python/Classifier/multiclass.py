import numpy
import pandas
from gravity import gravity_extraction
from windowing import windowing_multiclass
import double_classifier

def add_labels(X):
    training = X['Training']
    exercise = numpy.zeros(X.shape[0])
    i = 1
    current = 1

    while i < X.shape[0]:
        if training[i] == 1:
            exercise[i] = current
        else:
            if training[i-1] == 1:
                current = current+1
        i = i + 1

    X['Exercise'] = exercise
    return X

def dataframe_creation(X):
    data = {'Time': X['Time'],
            'xaa': X['aX_Ankle'], 'yaa': X['aY_Ankle'], 'zaa': X['aZ_Ankle'],
            'xag': X['gX_Ankle'], 'yag': X['gY_Ankle'], 'zag': X['gZ_Ankle'],
            'xwa': X['aX_Wrist'], 'ywa': X['aY_Wrist'], 'zwa': X['aZ_Wrist'],
            'xwg': X['gX_Wrist'], 'ywg': X['gY_Wrist'], 'zwg': X['gZ_Wrist']}
    df = pandas.DataFrame(data = data)

    label = X['Exercise']
    training = numpy.zeros(len(label))

    for i in range(len(label)):
        if label[i]:
            training[i] = 1

    df['Training'] = training

    return df

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

def classifier(dataFrame):
    # Creation of the dataframe
    dataFrame = dataframe_creation(dataFrame)

    # Gravity extraction
    dataFrame = gravity_extraction(dataFrame)

    # Labelling the exercise sessions
    dataFrame = add_labels(dataFrame)
    training = dataFrame['Training']

    # Windowing and features extraction
    dataFrame_windowed = windowing_multiclass(dataFrame)

    # Classification of the exercise session
    predicted_labels = double_classifier.classifier(dataFrame_windowed)

    # Final result (CSV format)
    label = numpy.zeros(dataFrame.shape[0])
    i = 1
    j = 0

    while i < dataFrame.shape[0]:
        if training[i] == 1:
            label[i] = predicted_labels[j]
        else:
            if training[i - 1] == 1:
                j = j + 1
        i = i + 1

    Label = []
    for i in range(len(label)):
        if label[i] == 0:
            Label.append('Rest')
        if label[i] == 1:
            Label.append('Push Press')
        if label[i] == 2:
            Label.append('Shoulder Press')
        if label[i] == 3:
            Label.append('Push Jerk')
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
