import numpy
import pandas

def windowing_binary(data):
    cols = data.columns.tolist()
    dataWindowed = pandas.DataFrame(columns=cols[:-1])

    # Sliding window approach
    length = 10  # Length of the window (in seconds)
    slide = 0.2  # Slide between consecutive windows (in seconds)

    i = 0
    while i + length * 50 < len(data.index):
        newRow = pandas.DataFrame(columns=cols[:-1])
        for column in cols[:-1]:
            window = numpy.array(data[column][int(i):int(i + length * 50)])
            newRow.loc[0, column] = window
        dataWindowed = dataWindowed.append(newRow, ignore_index=True)
        i += slide * 50

    return dataWindowed

def windowing_multiclass(final_df):

    # tengo traccia solo dell'allenamento
    training = final_df['Training'] == 1
    # salvo in un vettore le label delle sessioni
    exercises = final_df['Exercise']
    # droppo le colonne che non mi servono (rimane solo la parte relativa ai segnali)
    final_df.drop('Time', inplace = True, axis = 1)
    final_df.drop('Training', inplace = True, axis = 1)
    final_df.drop('Exercise', inplace = True, axis = 1)
    # per comodità trasformo il dataframe in una matrice (meglio per estrarre le features)
    total_matrix = final_df.to_numpy()
    # considero solo le righe che hanno esercizi
    total_matrix = total_matrix[training,]
    exercises = exercises[training]

    # Ciclo for esterno per i vari esercizi: creo una lista di dataframes che avrà tanti elementi
    # quanti sono gli esercizi della sessione, ciascuno sarà il dataframe completo di tutte le finestre
    # relative allo stesso esercizio.
    all_exercises_df = []

    for j in range(int(numpy.max(exercises))):

        # estraggo le righe corrispondenti alla fase di esercizio corrente
        indexes = exercises == j + 1
        # estraggo solo le righe che mi interessano (cioè esercizio corrente)
        matrix = total_matrix[indexes,:]

        # fisso i parametri che definiscono le finestre
        f_sample = 50 # frequenza di campionamento
        w_length = 5 # lunghezza finestra (s)
        w_offset = 1 # traslazione di ogni finestra dalla precedente (s)
        w_samples = w_length * f_sample # numero di campioni per ogni finestra
        begin_w = 0 # inizio di una finestra
        end_w = begin_w + w_samples # fine di una finestra

        # Preparo una lista vuota: sarà una lista di dataframes, uno per ciascuna finestra.
        # Ogni dataframe conterrà le features estratte per ogni finestra.
        all_windows_df = []

        # Ciclo while in cui scorro ogni finestra e chiamo le funzioni che estraggono le features (vedi sopra).
        # Poi unisco per riga tutti i vettori di output e i nomi per creare un unico grande vettore; lo trasformo
        # in dataframe e lo appendo alla lista all_windows_df
        while end_w < matrix.shape[0]:

            signal = matrix[begin_w:end_w,:]
            # calcolo anche la norma L^2 del segnale
            norm_signal = numpy.zeros((signal.shape[0], 6))
            for i in range(6):
                norm_signal[:,i] = numpy.sqrt(signal[:,3*i]**2 + signal[:,3*i+1]**2 + signal[:,3*i+2]**2)

            import utilities

            # media dei 18 segnali
            means, col_names_mean = utilities.mean(signal)
            # energia dei 18 segnali (diagonale matrice di covarianza)
            vars, col_names_covariance = utilities.covariance(signal)
            # correlazione tra i segnali (triangolo superiore matrice 12x12, escludendo la gravità)
            corrs, col_names_correlation = utilities.correlation(signal)
            # lunghezza della curva per i 18 segnali
            lengths, col_names_curve_length = utilities.curve_length(signal)
            # numero di zero-crossing dei segnali relativi alla gravità
            gcross, col_names_gzero_crossing = utilities.gzero_crossing(signal)
            # integrale della PSD della norma dei segnali (6 colonne) in 5 intervalli 0-25Hz
            integrals, col_names_PSD_integral = utilities.PSD_integral(norm_signal)
            # massimo valore in modulo per i 18 segnali
            max_vals, col_names_max_abs = utilities.max_abs(signal)
            # range per i 18 segnali
            ranges, col_names_srange = utilities.srange(signal)

            row_vector = numpy.concatenate((means,vars,corrs,lengths,gcross,integrals,max_vals,ranges), axis = 1)
            col_names = numpy.concatenate((col_names_mean,col_names_covariance,col_names_correlation,col_names_curve_length,col_names_gzero_crossing,col_names_PSD_integral,col_names_max_abs,col_names_srange))
            row_df = pandas.DataFrame(columns = col_names, data = row_vector)
            all_windows_df.append(row_df)

            # passo alla finestra successiva
            begin_w = begin_w + (w_offset * f_sample)
            end_w = begin_w + w_samples

        # Ora che ho la lista di dataframe completa per ogni finestra, unisco tutto nell'unico dataframe finale.
        # Aggiungo anche la label corrispondente all'esercizio corrente e appendo alla lista esterna
        features = pandas.concat(all_windows_df)
        features['Exercise'] = j + 1
        all_exercises_df.append(features)

    # Infine concateno i dataframe finali, ciascuno per il suo esercizio
    final_features = pandas.concat(all_exercises_df)

    return final_features
