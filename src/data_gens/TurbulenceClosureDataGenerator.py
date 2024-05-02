import numpy as np
try:
    import torch
    TORCH_EXISTS = True
except ImportError:
    TORCH_EXISTS = False

import typing as t


'''
    This class generates 3 term turbulence closure data for neural network training using solutions from
    "Turbulence closure modeling with data-driven techniques: Investigation of generalizable deep neural
    networks" by Taghizadeh, et al. (2021).
'''
class TurbulenceClosureDataGenerator:

    def __init__(self, model:str='LRR', type='numpy') -> None:
        self.LRR_C = np.array([3, 0, 0.8, 1.75, 1.31])
        self.SSG_C = np.array([3.4, 1.8, 0.36, 1.25, 0.4])
        self.generator = None
        self.mode = type

        if model.lower() == 'lrr':
            self.C = self.LRR_C
            self.generator = self.get_arsm_target_np if type.lower() == 'numpy' else self.get_arsm_target_torch
        elif model.lower() == 'ssg':
            self.C = self.SSG_C
            self.generator = self.get_arsm_target_np if type.lower() == 'numpy' else self.get_arsm_target_torch
        else: # SZL model
            self.generator = self.get_szl_target_np if type.lower() == 'numpy' else self.get_szl_target_torch

        # Calcualating L constants because they are only C dependent
        if model.lower() != 'szl':
            self.L_1_0 = self.C[0]/2 - 1
            self.L_1_1 = self.C[1] + 2
            self.L_2 = self.C[2]/2 - 2/3
            self.L_3 = self.C[3]/2 - 1
            self.L_4 = self.C[4]/2 - 1 
            if type.lower() == 'torch' and TORCH_EXISTS:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.L_1_0 = torch.tensor([self.L_1_0], dtype=torch.float64).to(self.device)
                self.L_1_1 = torch.tensor([self.L_1_1], dtype=torch.float64).to(self.device)
                self.L_2 = torch.tensor([self.L_2], dtype=torch.float64).to(self.device)
                self.L_3 = torch.tensor([self.L_3], dtype=torch.float64).to(self.device)
                self.L_4 = torch.tensor([self.L_4], dtype=torch.float64).to(self.device)



    def __call__(self, eta1: np.ndarray, eta2: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        if self.mode.lower() == 'numpy':
            return np.stack((eta1, eta2), axis=1), self.generator(eta1, eta2)
        else:
            eta1 = torch.tensor(eta1, dtype=torch.float64).to(self.device)
            eta2 = torch.tensor(eta2, dtype=torch.float64).to(self.device)
            return torch.stack((eta1, eta2), dim=1).cpu().detach().numpy(), self.generator(eta1, eta2).cpu().detach().numpy()


    '''
        Helper function for finding vectorizing determination of G_1, G_2, and G_3 for SZL model in numpy
    ''' 
    def get_szl_target_np(self, eta1, eta2) -> np.ndarray:
        sqrt_eta1 = np.sqrt(2*eta1)
        sqrt_eta2 = np.sqrt(2*eta2)

        G_1 = (-2/3)/(1.25 + sqrt_eta1 + .9*sqrt_eta2)
        G_2 = (-15/2)/(1000 + sqrt_eta1**3)
        G_3 = (3/2)/(1000 + sqrt_eta1**3)

        return np.stack((G_1, G_2, G_3), axis=1)


    '''
        Helper function for finding vectorizing determination of G_1, G_2, and G_3 Algebraic Reynolds Stress Model in numpy
    '''
    def get_arsm_target_np(self, eta1, eta2) -> np.ndarray:
        q = (self.L_1_0**2 + eta1*self.L_1_1*self.L_2 - (2/3)*eta1*(self.L_3**2) + 2*eta2*(self.L_4**2))/((eta1*self.L_1_1)**2)
        p = -(2*self.L_1_0)/(eta1*self.L_1_1)
        r = -(self.L_1_0*self.L_2)/((eta1*self.L_1_1)**2)

        a = q - (p**2)/3
        b = (1/27)*(2*p**3 - 9*p*q + 27*r)
        discriminant = (b**2)/4 + (a**3)/27

        # Stopping imaginary numbers in theta calculation for values of a that wont ever be used (d>=0)
        a[np.where(discriminant >= 0)] = -1

        theta = np.arccos((-b/2)/np.sqrt(-a**3/27))

        G_1 = np.zeros(len(eta1))

        # Case 1: discriminant >= 0
        d_pos = np.where(discriminant >= 0)
        G_1[d_pos] = -(p[d_pos]/3) + np.cbrt(-(b[d_pos]/2) + np.sqrt(discriminant[d_pos])) + np.cbrt(-(b[d_pos]/2) - np.sqrt(discriminant[d_pos]))

        # Case 2: discriminant < 0 and b < 0
        d_neg_b_neg = (discriminant < 0) & (b < 0)
        G_1[d_neg_b_neg] = -(p[d_neg_b_neg]/3) + 2*np.sqrt(-a[d_neg_b_neg]/3)*np.cos(theta[d_neg_b_neg]/3)

        # Case 3: discriminant < 0 and b >= 0
        d_neg_b_pos = (discriminant < 0) & (b >= 0)
        G_1[d_neg_b_pos] = -(p[d_neg_b_pos]/3) + 2*np.sqrt(-a[d_neg_b_pos]/3)*np.cos(theta[d_neg_b_pos]/3 + 2*np.pi/3)

        G_2 = (-self.L_4 * G_1)/(self.L_1_0 - eta1*self.L_1_1*G_1)
        G_3 = (2*self.L_3*G_1)/(self.L_1_0 - eta1*self.L_1_1*G_1)

        return np.stack((G_1, G_2, G_3), axis=1)
   
    '''
        Helper function for finding vectorizing determination of G_1, G_2, and G_3 for SZL model in torch
    ''' 
    def get_szl_target_torch(self, eta1, eta2):
        sqrt_eta1 = torch.sqrt(2*eta1)
        sqrt_eta2 = torch.sqrt(2*eta2)

        G_1 = (-2/3)/(1.25 + sqrt_eta1 + .9*sqrt_eta2)
        G_2 = (-15/2)/(1000 + sqrt_eta1**3)
        G_3 = (3/2)/(1000 + sqrt_eta1**3)

        return torch.stack((G_1, G_2, G_3), dim=1)

    '''
        Helper function for finding vectorizing determination of G_1, G_2, and G_3 Algebraic Reynolds Stress Model in PyTorch
    '''
    def get_arsm_target_torch(self, eta1, eta2):
        q = (self.L_1_0**2 + eta1*self.L_1_1*self.L_2 - (2/3)*eta1*(self.L_3**2) + 2*eta2*(self.L_4**2))/((eta1*self.L_1_1)**2)
        p = -(2*self.L_1_0)/(eta1*self.L_1_1)
        r = -(self.L_1_0*self.L_2)/((eta1*self.L_1_1)**2)

        a = q - (p**2)/3
        b = (1/27)*(2*p**3 - 9*p*q + 27*r)
        discriminant = (b**2)/4 + (a**3)/27

        # Stopping imaginary numbers in theta calculation for values of a that wont ever be used (d>=0)
        a[torch.where(discriminant >= 0)] = -1

        theta = torch.arccos((-b/2)/torch.sqrt(-a**3/27))

        G_1 = torch.zeros(len(eta1), dtype=torch.float64).to(self.device)

        # Case 1: discriminant >= 0 (funky stuff because torch doesn't have a cube root for negative numbers)
        d_pos = torch.where(discriminant >= 0)
        inner_p = -(b[d_pos]/2) + torch.sqrt(discriminant[d_pos])
        inner_m = -(b[d_pos]/2) - torch.sqrt(discriminant[d_pos])
        G_1[d_pos] = -(p[d_pos]/3) + inner_p.sign()*torch.pow(inner_p.abs(), (1/3)) + inner_m.sign()*torch.pow(inner_m.abs(), (1/3))

        # Case 2: discriminant < 0 and b < 0
        d_neg_b_neg = (discriminant < 0) & (b < 0)
        G_1[d_neg_b_neg] = -(p[d_neg_b_neg]/3) + 2*torch.sqrt(-a[d_neg_b_neg]/3)*torch.cos(theta[d_neg_b_neg]/3)

        # Case 3: discriminant < 0 and b >= 0
        d_neg_b_pos = (discriminant < 0) & (b >= 0)
        G_1[d_neg_b_pos] = -(p[d_neg_b_pos]/3) + 2*torch.sqrt(-a[d_neg_b_pos]/3)*torch.cos(theta[d_neg_b_pos]/3 + 2*torch.pi/3)

        G_2 = (-self.L_4 * G_1)/(self.L_1_0 - eta1*self.L_1_1*G_1)
        G_3 = (2*self.L_3*G_1)/(self.L_1_0 - eta1*self.L_1_1*G_1)

        return torch.stack((G_1, G_2, G_3), dim=1)
    


    