import pandas
import numpy
import scipy
import scipy.integrate

def mean(signal): # media dei segnali x,y,z per i 4 sensori + 2 gravità (18 features)

    means = signal.mean(0) # media di ciascuna colonna
    means = numpy.reshape(means, (1, means.shape[0]))
    col_names_mean = ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13','m14','m15','m16','m17','m18']
    return means, col_names_mean


def covariance(signal): # matrice di covarianza 18x18 (per ora solo la diagonale -> 18 features)

    # devo trasformare la matrice in un dataframe
    cols = ['xaa','yaa','zaa','gax','gay','gaz','xag','yag','zag','xwa','ywa','zwa','gwx','gwy','gwz','xwg','ywg','zwg']
    signal_df = pandas.DataFrame(columns = cols, data = signal)
    cov_matrix = signal_df.cov()
    # mi riporto alla forma di una matrice
    cov_matrix = cov_matrix.to_numpy()
    vars = cov_matrix.diagonal()
    vars = numpy.reshape(vars, (1, vars.shape[0]))
    col_names_covariance = ['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13','var14','var15','var16','var17','var18']
    return vars, col_names_covariance


def norm_covariance(norm_signal): # la lascio qua ma in realtà non viene usata...

    # devo trasformare la matrice in un dataframe
    cols = ['aa','ag','wa','wg']
    norm_signal_df = pandas.DataFrame(columns = cols, data = norm_signal)
    cov_matrix = norm_signal_df.cov()
    # mi riporto alla forma di una matrice
    cov_matrix = cov_matrix.to_numpy()
    norm_vars = cov_matrix.diagonal()
    norm_vars = numpy.reshape(norm_vars, (1, norm_vars.shape[0]))
    col_names_norm_covariance = ['nvar1','nvar2','nvar3','nvar4']
    return norm_vars, col_names_norm_covariance


def correlation(signal): # matrice di correlazione 12x12 dei segnali senza gravità (triangolo superiore escludendo la diagonale -> 66 features)

    # devo trasformare la matrice in un dataframe
    cols = ['xaa','yaa','zaa','gax','gay','gaz','xag','yag','zag','xwa','ywa','zwa','gwx','gwy','gwz','xwg','ywg','zwg']
    signal_df = pandas.DataFrame(columns = cols, data = signal)
    # elimino le colonne corrispondenti ai segnali di gravità
    signal_df.drop('gax', inplace = True, axis = 1)
    signal_df.drop('gay', inplace = True, axis = 1)
    signal_df.drop('gaz', inplace = True, axis = 1)
    signal_df.drop('gwx', inplace = True, axis = 1)
    signal_df.drop('gwy', inplace = True, axis = 1)
    signal_df.drop('gwz', inplace = True, axis = 1)
    corr_matrix = signal_df.corr()
    # mi riporto alla forma di una matrice
    corr_matrix = corr_matrix.to_numpy()
    corrs = corr_matrix[numpy.triu_indices(corr_matrix.shape[0], k = 1)]
    corrs = numpy.reshape(corrs, (1, corrs.shape[0]))
    # NB: i numeri si riferiscono alla matrice 12x12, non a quella 18x18!!
    col_names_correlation = ['corr12','corr13','corr14','corr15','corr16','corr17','corr18','corr19','corr110','corr111','corr112','corr23','corr24','corr25','corr26','corr27','corr28','corr29','corr210','corr211','corr212',\
                             'corr34','corr35','corr36','corr37','corr38','corr39','corr310','corr311','corr312','corr45','corr46','corr47','corr48','corr49','corr410','corr411','corr412','corr56','corr57','corr58','corr59','corr510','corr511','corr512',\
                             'corr67','corr68','corr69','corr610','corr611','corr612','corr78','corr79','corr710','corr711','corr712','corr89','corr810','corr811','corr812','corr910','corr911','corr912','corr1011','corr1012','corr1112']
    return corrs, col_names_correlation


def curve_length(signal): # lunghezza della curva dei 12 segnali x,y,z dei 4 sensori (12 features)

    lengths = numpy.sum(numpy.abs(signal[1:signal.shape[0],0:18] - signal[0:signal.shape[0]-1,0:18]), 0)
    lengths = numpy.reshape(lengths, (1, lengths.shape[0]))
    col_names_curve_length = ['l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12','l13','l14','l15','l16','l17','l18']
    return lengths, col_names_curve_length


def gzero_crossing(signal): # numero di zero-crossing dei 6 segnali di gravità (6 features)

    # estraggo la sotto matrice corrispondente ai 6 segnali di gravità
    gravity = signal[:, [3,4,5,12,13,14]]
    gcross = (numpy.diff(numpy.sign(gravity), axis = 0) != 0).sum(axis = 0)
    gcross = numpy.reshape(gcross, (1, gcross.shape[0]))
    col_names_gzero_crossing = ['gc1','gc2','gc3','gc4','gc5','gc6']
    return gcross, col_names_gzero_crossing


def PSD_integral(norm_signal): # calcolo la PSD delle 6 colonne e integro su 5 intervalli in 0-25Hz (30 features)

    # definisco il vettore di frequenze (che sarà sempre uguale)
    freq = numpy.fft.fftfreq(norm_signal.shape[0]) * 50
    # estraggo la porzione di frequenze umane (primi 125 elementi)
    human_freq = freq[1:126]
    # creo una lista che conterrà tutte le mie features (alla fine farò un reshape)
    integrals = []
    # ciclo for su tutte le colonne (le norme dei 6 segnali)
    for col in range(6):
        PSD = numpy.abs(numpy.fft.fft(norm_signal[:,col]))**2
        for i in [0,25,50,75,100]:
            x = human_freq[i:i+25]
            y = PSD[i:i+25]
            integral = scipy.integrate.simps(y,x)
            integrals.append(integral)
    # trasformo la lista in una matrice riga e creo la lista dei nomi colonna
    integrals = numpy.reshape(integrals, (1, len(integrals)))
    col_names_PSD_integral = ['int1','int2','int3','int4','int5','int6','int7','int8','int9','int10',\
                              'int11','int12','int13','int14','int15','int16','int17','int18','int19','int20',\
                              'int21','int22','int23','int24','int25','int26','int27','int28','int29','int30']
    return integrals, col_names_PSD_integral


def max_abs(signal): # valore massimo in modulo dei segnali (18 features)

    max_vals = numpy.abs(signal).max(0)
    max_vals = numpy.reshape(max_vals, (1, len(max_vals)))
    col_names_max_abs = ['max1','max2','max3','max4','max5','max6','max7','max8','max9','max10','max11','max12','max13','max14','max15','max16','max17','max18']
    return max_vals, col_names_max_abs


def srange(signal): # range di ciascun segnale (valore assoluto di max-min) (18 features)

    max_vals = signal.max(0)
    min_vals = signal.min(0)
    ranges = numpy.abs(max_vals - min_vals)
    ranges = numpy.reshape(ranges, (1, len(ranges)))
    col_names_srange = ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18']
    return ranges, col_names_srange
