import pandas
import numpy
import scipy.signal


# Helper function: given a signal it returns the (normed) autocorrelation
def autocorrelation(signal):
    x = numpy.correlate(signal, signal, mode="full")
    x = x[x.size // 2:]
    result = x / float(x.max())
    return result


# Helper function: compute bandpower
def bandpower(signal, fmin, fmax):
    f, spectral_density = scipy.signal.periodogram(signal, 50)
    ind_min = numpy.argmax(f > fmin) - 1
    ind_max = numpy.argmax(f > fmax) - 1
    return numpy.trapz(spectral_density[ind_min: ind_max], f[ind_min: ind_max])


def feature_extractor(signal):
    result = list()
    signal_ac = autocorrelation(signal)

    # Autocorrelation-realted features: number of peaks, prominent peaks and weak peaks, height of the first peak
    # after 0 and maximum height of the autocorrelation function
    n_peaks = len(scipy.signal.find_peaks(signal_ac)[0])
    result.append(n_peaks)
    n_prom_peaks = len(scipy.signal.find_peaks(signal_ac, prominence=0.17, distance=100)[0])
    result.append(n_prom_peaks)
    n_weak_peaks = len(scipy.signal.find_peaks(signal_ac, prominence=[0, 0.17], wlen=200)[0])
    result.append(n_weak_peaks)
    max_ac = max(signal_ac[1:])
    result.append(max_ac)
    try:
        max_peak = scipy.signal.find_peaks(signal_ac)[0][1]
    except:
        max_peak = 0
    result.append(max_peak)

    # Band Powers (10 features)
    bands_f = numpy.linspace(0.1, 25, 11)
    bp = list()
    for index in range(10):
        band = bandpower(signal, bands_f[index], bands_f[index + 1])
        bp.append(band)
    result = result + bp

    # Mean, standard deviation and variance
    mean = numpy.mean(signal)
    result.append(mean)
#    std = numpy.std(signal)
#    result.append(std)
    variance = numpy.var(signal)
    result.append(variance)

    # RMS amplitude
    f, spectral_density = scipy.signal.periodogram(signal, 50)
    rms = numpy.sqrt(spectral_density.max())
    result.append(rms)

    # RMS amplitude after cumulative summation (acceleration -> velocity)
    signal_cumsum = numpy.cumsum(signal)
    f, spectral_density = scipy.signal.periodogram(signal_cumsum, 50)
    rms_cumsum = numpy.sqrt(spectral_density.max())
    result.append(rms_cumsum)

    return result


def features(data):
    # Creating an empty dataframe with the right column names
#    feats = ["nPeaks", "nPromPeaks", "nWeakPeaks", "maxAc", "maxPeak", "BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7",
#             "BP8", "BP9", "BP10", "mean", "std", "variance", "rms", "rms_cumsum", "mean1", "std1", "var1", "rms1",
#             "mean2", "std2", "var2", "rms2"]

    feats = ["nPeaks", "nPromPeaks", "nWeakPeaks", "maxAc", "maxPeak", "BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7",
             "BP8", "BP9", "BP10", "mean", "variance", "rms", "rms_cumsum"]
    names = ["aX_Wrist", "aY_Wrist", "aZ_Wrist", "gX_Wrist", "gY_Wrist", "gZ_Wrist", "aX_Ankle", "aY_Ankle", "aZ_Ankle",
             "gX_Ankle", "gY_Ankle", "gZ_Ankle"]
    cols = []
    for name in names:
        for feat in feats:
            cols.append(name + feat)
    features = pandas.DataFrame(columns=cols)

    # Creating the rows of the dataframe
    for i in range(len(data.index)):
        newRow = []
        for column in data.columns:
            newRow = newRow + feature_extractor(data[column][i])
        features.loc[i] = newRow

    return features
