import numpy

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
        i = i+1

    X['Exercise'] = exercise

    return X
