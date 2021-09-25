import numpy
import pandas
import utilities

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
    training = final_df['Training'] == 1 # Consider only training sessions
    exercises = final_df['Exercise']
    final_df.drop('Time', inplace = True, axis = 1)
    final_df.drop('Training', inplace = True, axis = 1)
    final_df.drop('Exercise', inplace = True, axis = 1)
    total_matrix = final_df.to_numpy()
    total_matrix = total_matrix[training,] # Select only exercise rows
    exercises = exercises[training]

    all_exercises_df = []
    for j in range(int(numpy.max(exercises))):
        indexes = exercises == j + 1
        matrix = total_matrix[indexes,:]

        f_sample = 50 # sampling frequency
        w_length = 5 # window length
        w_offset = 1 # window slide
        w_samples = w_length * f_sample # number of samples per window
        begin_w = 0 # start of the window
        end_w = begin_w + w_samples # end of the window

        all_windows_df = []
        while end_w < matrix.shape[0]:
            signal = matrix[begin_w:end_w,:]
            norm_signal = numpy.zeros((signal.shape[0], 6)) # L2 norm
            for i in range(6):
                norm_signal[:,i] = numpy.sqrt(signal[:,3*i]**2 + signal[:,3*i+1]**2 + signal[:,3*i+2]**2)

            # mean
            means, col_names_mean = utilities.mean(signal)
            # diagonal elements of covariance matrix
            vars, col_names_covariance = utilities.covariance(signal)
            # correlation of each pair of signals
            corrs, col_names_correlation = utilities.correlation(signal)
            # curve length
            lengths, col_names_curve_length = utilities.curve_length(signal)
            # number of zero-crossing (gravity)
            gcross, col_names_gzero_crossing = utilities.gzero_crossing(signal)
            # power spectral density integration (band power computation)
            integrals, col_names_PSD_integral = utilities.PSD_integral(norm_signal)
            # max value of the signals
            max_vals, col_names_max_abs = utilities.max_abs(signal)
            # range of the signals
            ranges, col_names_srange = utilities.srange(signal)

            row_vector = numpy.concatenate((means,vars,corrs,lengths,gcross,integrals,max_vals,ranges), axis = 1)
            col_names = numpy.concatenate((col_names_mean,col_names_covariance,col_names_correlation,col_names_curve_length,col_names_gzero_crossing,col_names_PSD_integral,col_names_max_abs,col_names_srange))
            row_df = pandas.DataFrame(columns = col_names, data = row_vector)
            all_windows_df.append(row_df)

            # update to next window
            begin_w = begin_w + (w_offset * f_sample)
            end_w = begin_w + w_samples

        features = pandas.concat(all_windows_df)
        features['Exercise'] = j + 1
        all_exercises_df.append(features)

    final_features = pandas.concat(all_exercises_df)
    return final_features
