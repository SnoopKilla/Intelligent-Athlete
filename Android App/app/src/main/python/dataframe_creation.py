import pandas
import numpy

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
