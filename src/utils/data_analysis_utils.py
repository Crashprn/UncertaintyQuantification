import numpy as np
from scipy.stats import norm


'''
Function that calculates the sharpness score of a prediction.
Where the sharpness score is from
"Methods for comparing uncertainty quantifications for material
property predictions" by Tran K. et al.

Parameters:
y_pred_std: np.ndarray
    The array of standard deviations for each prediction.
axis_mean: int, optional
    The axis to calculate the mean of the variance.
    Default is None.

Returns:
np.ndarray
    The sharpness score of the prediction.
'''
def sharpness_score(y_pred_std, axis_mean=None):
    return np.mean(y_pred_std**2, axis=axis_mean)


'''
Function that calculates the coefficient of variation of a prediction.
Where the coefficient of variation is from
"Methods for comparing uncertainty quantifications for material
property predictions" by Tran K. et al.

Parameters:
y_pred_std: np.ndarray
    The array of standard deviations for each prediction.
axis_mean: int, optional
    The axis to calculate the mean of the variance.
    Default is None.

Returns:
np.ndarray
    The coefficient of variation of the prediction.
'''
def coefficient_of_variation(y_pred_std, axis_mean=None):
    mean_std = np.mean(y_pred_std, axis=axis_mean)
    return np.sqrt(np.sum((y_pred_std-mean_std)**2, axis=axis_mean)/(y_pred_std.shape[axis_mean]-1)) / mean_std


'''
Function to calculate the negative log likelihood of IID data using a Multivariate
Gaussian distribution.

Parameters:
y_true: np.ndarray
    The true values of the data.
y_pred: np.ndarray
    The predicted values of the data.
axis_samples: int, optional
    The axis that represents the samples.
    Default is 0.
axis_points: int, optional
    The axis that represents the data point dimension.
    Default is 1.
axis_features: int, optional
    The axis that represents the features dimension.
    Default is 2.

Returns:
float
    The negative log likelihood of the data.
'''
def negative_log_likelihood(y_true, y_pred_mean, y_pred_cov, axis_features=1):
    
    reg = np.expand_dims(y_true - y_pred_mean, axis=1)
    trans = reg.transpose((0, 2, 1))

    quadratic = (reg @ np.linalg.inv(y_pred_cov) @ trans).squeeze()
    det = np.linalg.det(y_pred_cov)
    log_pi = y_pred_mean.shape[axis_features]*np.log(2*np.pi)

    return (0.5*(quadratic + np.log(det) + log_pi)).sum()

'''
Calculating the calibration error of a multi-output prediction by comparing 
with a MultiVariate Gaussian distribution.

Parameters:
y_true: Data x Features np.ndarray 
    The true values of the data.
y_pred: Samples x Data x Features np.ndarray
    The predicted values of the data.
num_bins: int, optional
    The number of bins to use for the calibration curve.
    Default is 100.

Returns:
np.ndarray, np.ndarray
    The calibration curve of the prediction in the form (gaussian_cdf, empirical_cdf).
'''
def calibration_curve(y_true, y_pred, num_bins=100):
    y_pred_mean = y_pred.mean(axis=0)
    y_pred_std = y_pred.std(axis=0)
    p_i_arr = np.linspace(0, 1, num_bins+1)
    p_i_hat = np.zeros(p_i_arr.shape[0])

    for i, p_i in enumerate(p_i_arr):
        y_i = norm.ppf(p_i)
        y_i = y_i * y_pred_std + y_pred_mean

        emp_cdf_y_i = np.mean(np.prod((y_pred <= y_i), axis=2), axis=0)
        p_i_hat[i] = np.mean(emp_cdf_y_i)
    
    return (p_i_arr, p_i_hat)


'''
Calculating the calibration error of a multi-output prediction by comparing
the assumed cdf and emperical cdf.
Found in "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
by V. Kuleshov , N. Fenner and S. Ermon

Parameters:
p_i_arr: np.ndarray
    The assumed cdf of the prediction.
p_i_hat: np.ndarray
    The empirical cdf of the prediction.

Returns:
float
    The calibration error of the prediction.
'''
def calibration_error(p_i_arr, p_i_hat):
    return np.mean((p_i_arr - p_i_hat)**2)


'''
Calculating the miscalibration area of a multi-output prediction by comparing
the assumed cdf and emperical cdf.
Found in "Methods for comparing uncertainty quantifications for material
property predictions" by Tran K. et al.

Parameters:
p_i_arr: np.ndarray
    The assumed cdf of the prediction.
p_i_hat: np.ndarray
    The empirical cdf of the prediction.
num_bins: int, optional
    The number of bins to use for the calibration curve so dx = 1/num_bins.
    Default is 100.

Returns:
float
    The miscalibration area of the prediction.
'''
def miscalibration_area(p_i_arr, p_i_hat, num_bins=100):
    return np.trapz(np.abs(p_i_arr - p_i_hat), dx=1/(num_bins))


'''
Helper function to calculate the arrray of covariance matrices for a multi-output prediction.

Parameters:
y_pred: np.ndarray
    The predicted values of the data.
axis_samples: int, optional
    The axis that represents the samples.
    Default is 0.
axis_points: int, optional
    The axis that represents the data point dimension.
    Default is 1.
axis_features: int, optional
    The axis that represents the features dimension.
    Default is 2.

Returns:
np.ndarray
    The array of covariance matrices for the prediction.

'''
def covariance_matrix(y_pred, axis_samples=0, axis_points=1, axis_features=2, ddof=1):
    y_pred_mean = y_pred.mean(axis=axis_samples)
    diff = y_pred - y_pred_mean

    cov =  np.matmul(diff.transpose((axis_points, axis_features, axis_samples)), diff.transpose((axis_points, axis_samples, axis_features))) / (y_pred.shape[axis_samples]-ddof)

    return cov


