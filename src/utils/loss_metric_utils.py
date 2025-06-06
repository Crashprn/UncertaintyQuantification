import torch
from torch import nn

'''
Calculating Root Mean Squared Error (RMSE) of a multi-output prediction.

Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The RMSE of the prediction.
'''
class RMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RMSELoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        l = torch.sqrt(torch.mean(torch.pow((y_true - y_pred), 2)))

        return l
    
'''
Calculating Mean Absolute Percentage Error (MAPE) of a multi-output prediction.
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The MAPE of the prediction.
'''
class MAPELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MAPELoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        l = torch.mean(torch.abs((y_true - y_pred) / (y_true + self.eta)))

        return l

'''
Calculating Mean Absolute Relative Percentage Deviation (MARPD) of a multi-output prediction.
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The MARPD of the prediction.
''' 
class MARPDLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MARPDLoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        l = torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred) + self.eta)))

        return l

'''
Calculating Median Absolute Error Loss of a multi-output prediction.
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The Median Absolute Error Loss of the prediction.
'''
class MedianAELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MedianAELoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        l = torch.median(torch.abs(y_true - y_pred))

        return l

'''
Calculating Mean Absolute Error Loss of a multi-output prediction.
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The Mean Absolute Error Loss of the prediction.
'''
class MeanAELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MeanAELoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        l = torch.mean(torch.abs(y_true - y_pred))

        return l

'''
Calculating R2 Loss of a multi-output prediction.
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    The predicted values of the data.

Returns:
float
    The R2 Loss of the prediction.
'''
class R2Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(R2Loss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum(torch.pow(y_true - y_mean, 2))
        ss_res = torch.sum(torch.pow(y_true - y_pred, 2))

        r2 = 1 - ss_res / (ss_tot + self.eta)

        return r2

'''
Calculating the Negative Log Likelihood Loss of a Diagonal Multivariate Gaussian distribution
Parameters:
y_true: torch.Tensor
    The true values of the data.
y_pred: torch.Tensor
    (mean, std) of the predicted values of the data.

Returns:
float
    The Negative Log Likelihood Loss of the prediction.
'''
class DiagNLLLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DiagNLLLoss, self).__init__()
        self.eta = 1e-8

    def forward(self, y_pred, y_true):
        y_predictions, y_sigmas = y_pred 

        # Returns log(sigma^2)
        #l1 = y_predictions.shape[0] * y_sigmas.sum() + torch.sum(torch.pow((y_true - y_predictions), 2) / torch.exp(y_sigmas))
        l2 = torch.log(y_sigmas).sum()*y_predictions.shape[0] + torch.sum(torch.pow((y_true - y_predictions), 2)/ y_sigmas)

        return l2

class BetaNLL(nn.Module):
    def __init__(self, beta=1, *args, **kwargs):
        super(BetaNLL, self).__init__()
        self.beta = beta
        self.var_reg_lam = 0.1

    def forward(self, y_pred, y_true):
        y_predictions, y_variance = y_pred 

        loss = 0.5 * ((y_predictions - y_true) ** 2 / (y_variance) + 3*torch.log(y_variance))

        if self.beta > 0:
            loss = loss * (y_variance.detach() ** self.beta)

        return loss.sum()