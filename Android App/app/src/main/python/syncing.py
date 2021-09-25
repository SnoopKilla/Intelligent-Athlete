from datetime import datetime, timezone, timedelta
import pandas
from scipy.interpolate import interp1d
from scipy import signal

def str_to_ms(time_str):
    h, m, s, f = time_str.split(':')
    return int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + int(int(f) / 1000)

def synced(wA, wG, aA, aG):
    # Changing the names of the columns for the 4 dataframes so that they match
    wristAcc = pandas.read_csv(wA)
    wristAcc.rename(columns={"x-axis (g)": "x", "y-axis (g)": "y", "z-axis (g)": "z"}, inplace=True)
    wristGyr = pandas.read_csv(wG)
    wristGyr.rename(columns={"x-axis (deg/s)": "x", "y-axis (deg/s)": "y", "z-axis (deg/s)": "z"}, inplace=True)
    ankleAcc = pandas.read_csv(aA)
    ankleAcc.rename(columns={"x-axis (g)": "x", "y-axis (g)": "y", "z-axis (g)": "z"}, inplace=True)
    ankleGyr = pandas.read_csv(aG)
    ankleGyr.rename(columns={"x-axis (deg/s)": "x", "y-axis (deg/s)": "y", "z-axis (deg/s)": "z"}, inplace=True)
    dfs = [wristAcc, wristGyr, ankleAcc, ankleGyr]

    # Deleting useless columns and computing timestamp
    data = []
    for dataFrame in dfs:
        if dataFrame.columns[0] == "epoc (ms)":
            dataFrame.drop("epoc (ms)", inplace=True, axis=1)
        else:
            dataFrame.drop("epoch (ms)", inplace=True, axis=1)
        dataFrame.drop("elapsed (s)", inplace=True, axis=1)
        if dataFrame.columns[0] == "timestamp (+0200)":
            dataFrame = dataFrame.assign(timestamp=lambda df: df["timestamp (+0200)"].map(
                lambda time: str_to_ms(datetime.strptime(time[11:], "%H.%M.%S.%f").strftime("%H:%M:%S:%f"))))
            dataFrame.drop("timestamp (+0200)", inplace=True, axis=1)
        else:
            dataFrame = dataFrame.assign(timestamp=lambda df: df["time (01:00)"].map(
                lambda time: str_to_ms(datetime.strptime(time[11:], "%H:%M:%S.%f").strftime("%H:%M:%S:%f"))))
            dataFrame.drop("time (01:00)", inplace=True, axis=1)
        data.append(dataFrame)

    # Time vector for the interpolation: from maximum starting time to minimum ending time (100Hz sampling frequency
    start = []
    end = []
    for dataFrame in data:
        start.append(dataFrame["timestamp"].iloc[0])
        end.append(dataFrame["timestamp"].iloc[-1])
    start = max(start)
    end = min(end)
    timeInterval = range(start, end, 10)

    # Syncing and filtering the signals
    FiltNum = [0.1929,1.543,5.4005,10.8009,13.5011,10.8009,5.4005,1.543,0.1929]
    FiltDen = [1,4.7845,10.445,13.4577,11.1293,6.0253,2.0793,0.4172,0.0372]
    synced = []
    for dataFrame in data:
        x = list(dataFrame["x"])
        y = list(dataFrame["y"])
        z = list(dataFrame["z"])
        timestamp = list(dataFrame["timestamp"])
        fx = interp1d(timestamp, x)
        fy = interp1d(timestamp, y)
        fz = interp1d(timestamp, z)
        xNew = fx(timeInterval)
        xNew = signal.lfilter(FiltNum, FiltDen, xNew)
        yNew = fy(timeInterval)
        yNew = signal.lfilter(FiltNum, FiltDen, yNew)
        zNew = fz(timeInterval)
        zNew = signal.lfilter(FiltNum, FiltDen, zNew)
        dataSync = pandas.DataFrame({"Timestamp": timeInterval, "x": xNew, "y": yNew, "z": zNew})
        dataSync = dataSync.round({"x": 3, "y": 3, "z": 3})
        dataSync = dataSync.assign(Time=lambda df: df["Timestamp"].map(
            lambda timestamp: datetime.fromtimestamp(timestamp / 1000, timezone(timedelta(hours=0))).strftime(
                "%H:%M:%S.%f")[0:12]))
        dataSync.drop("Timestamp", inplace=True, axis=1)
        dataSync = dataSync[dataSync.index%2 == 0] # Undersampling
        synced.append(dataSync)

    # Merging the results in syncTotal
    synced[0].rename(columns={"x": "aX_Wrist", "y": "aY_Wrist", "z": "aZ_Wrist"}, inplace=True)
    synced[1].rename(columns={"x": "gX_Wrist", "y": "gY_Wrist", "z": "gZ_Wrist"}, inplace=True)
    synced[2].rename(columns={"x": "aX_Ankle", "y": "aY_Ankle", "z": "aZ_Ankle"}, inplace=True)
    synced[3].rename(columns={"x": "gX_Ankle", "y": "gY_Ankle", "z": "gZ_Ankle"}, inplace=True)
    syncTotal = pandas.merge(synced[0], synced[1], on="Time")
    syncTotal = pandas.merge(syncTotal, synced[2], on="Time")
    syncTotal = pandas.merge(syncTotal, synced[3], on="Time")
    cols = syncTotal.columns.tolist()
    cols = cols[0:3] + cols[4:] + [cols[3]]
    syncTotal = syncTotal[cols]

    return syncTotal
