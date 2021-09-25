import pandas
import numpy as np
from ahrs.filters import EKF

def gravity_extraction(synced_final):
    # CONSTANTS
    deg2rad = np.pi/180
    rad2deg = 1/deg2rad
    gravity = 9.81
    freq = 50

    # Estraggo e converto le unità di misura di giroscopio e accelerometro
    # Caviglia
    acc_cav = synced_final[['xaa','yaa','zaa']].to_numpy()*gravity
    gyr_cav = synced_final[['xag','yag','zag']].to_numpy()*deg2rad
    # Polso
    acc_pol = synced_final[['xwa','ywa','zwa']].to_numpy()*gravity
    gyr_pol = synced_final[['xwg','ywg','zwg']].to_numpy()*deg2rad

    # Creo oggetti ekf per polso e caviglia
    ekf_cav = EKF( gyr=gyr_cav, acc=acc_cav,  frequency = freq, frame='ENU')
    ekf_pol = EKF( gyr=gyr_pol, acc=acc_pol,  frequency = freq, frame='ENU')

    # Estraggo quaternioni (orientamento)
    Q_cav = ekf_cav.Q
    Q_pol = ekf_pol.Q

    # Estraggo componenti quaternioni per semplicità
    # Caviglia
    Qw_cav = Q_cav[:,0]
    Qx_cav = Q_cav[:,1]
    Qy_cav = Q_cav[:,2]
    Qz_cav = Q_cav[:,3]

    # Polso
    Qw_pol = Q_pol[:,0]
    Qx_pol = Q_pol[:,1]
    Qy_pol = Q_pol[:,2]
    Qz_pol = Q_pol[:,3]

    # Estraggo segnale di gravità
    # Caviglia
    gax = 2*(Qx_cav*Qz_cav-Qw_cav*Qy_cav)
    gay = 2*(Qw_cav*Qx_cav+Qy_cav*Qz_cav)
    gaz = 1-2*(Qx_cav**2+Qy_cav**2)

    # Polso
    gwx = 2*(Qx_pol*Qz_pol-Qw_pol*Qy_pol)
    gwy = 2*(Qw_pol*Qx_pol+Qy_pol*Qz_pol)
    gwz = 1-2*(Qx_pol**2+Qy_pol**2)

    # Separo user acceleration
    # Caviglia
    uax = acc_cav[:,0]/gravity-gax
    uay = acc_cav[:,1]/gravity-gay
    uaz = acc_cav[:,2]/gravity-gaz

    # Polso
    uwx = acc_pol[:,0]/gravity-gwx
    uwy = acc_pol[:,1]/gravity-gwy
    uwz = acc_pol[:,2]/gravity-gwz

    # Creo nuovo dataframe

    time = synced_final['Time']

    data = {'Time' : time,
            'xaa' : uax,          'yaa' : uay,          'zaa' : uaz,
            'gax' : gax,          'gay' : gay,          'gaz' : gaz,
            'xag' : gyr_cav[:,0], 'yag' : gyr_cav[:,1], 'zag' : gyr_cav[:,2],
            'xwa' : uwx,          'ywa' : uwy,          'zwa' : uwz,
            'gwx' : gwx,          'gwy' : gwy,          'gwz' : gwz,
            'xwg' : gyr_pol[:,0], 'ywg' : gyr_pol[:,1], 'zwg' : gyr_pol[:,2],
            'Training': synced_final['Training']}

    df_extracted = pandas.DataFrame(data)

    return df_extracted
