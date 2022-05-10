import os
import numpy as np

def sinusoidal_regression(temperature_timeseries, output=None):
    '''
    temperature_timeseries (ndim array) LST maps (°C) in chronological order stored in array of shape (8,y,x)
    output = 'all' to return all reuslts (e.g. Amplitude, RMSE, Offset)
    '''
    
    flights = np.array([9,11,13,15,17,19,21,22])
    
    #assuming the mean debris layer temperature to be the mean of LST and debris ice interface (0°C)
    Tdi = 273.15 # debris ice interface
    LST_in_K = temperature_timeseries+273.15 #°C in K
    
    MeanDebrisT = (LST_in_K+Tdi)/2
            
    def sine_fit(T_mean, times):
        
        period = 24
        omega = (2*np.pi)/period
        
        #design matrix 
        X=np.vstack([np.ones(times.shape),
                     np.sin(omega*times),
                     np.cos(omega*times)]).T
                     
        LeastSquaresSolution, residuals, _ , _ = np.linalg.lstsq(X, T_mean, rcond=None)
        
        a = LeastSquaresSolution[0]
        b = LeastSquaresSolution[1]
        c = LeastSquaresSolution[2]
        
        fitfunc = lambda t: a+b*np.sin(omega*t)+c*np.cos(omega*t)
        
        results = {"offset": a, 
                   "sin_amp": b, 
                   "cos_amp": c, 
                   "fitfunc": fitfunc, 
                   "residuals":residuals}
                   
        return results
        
    FittedSine=np.apply_along_axis(sine_fit, 0, np.nan_to_num(MeanDebrisT), times=flights)
    
    amplitudeSin = np.zeros(FittedSine.shape)
    amplitudeCos = np.zeros(FittedSine.shape)
    offset=np.zeros(FittedSine.shape)
    residuals=np.zeros(FittedSine.shape)
    
    for row in range(FittedSine.shape[0]):
        for col in range(FittedSine.shape[1]):
            amplitudeSin[row][col]=FittedSine[row][col]['sin_amp']
            amplitudeCos[row][col]=FittedSine[row][col]['cos_amp']
            offset[row][col]=FittedSine[row][col]['offset']
            residuals[row][col]=FittedSine[row][col]['residuals']
        
    rmse=np.sqrt(residuals/len(MeanDebrisT))
    
    period = 24
    omega = (2*np.pi)/period
    
    warming_rate = []
    for t in flights:
        first_derivative = -omega*(amplitudeCos*np.sin(omega*t)-amplitudeSin*np.cos(omega*t))
        wr = first_derivative/3600 #unit K/s   
        warming_rate.append(wr)
    
    warming_rate = np.array(warming_rate)
    
    result = {"wr": warming_rate, 
              "sin_amp": amplitudeSin, 
              "cos_amp": amplitudeCos, 
              "rmse": rmse, 
              "offset":offset}
                   
    if output == 'all':
        return result
    else:
        return warming_rate