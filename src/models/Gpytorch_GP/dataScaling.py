import torch

class transform():
    def __init__(self, inputs):
        self.inputs = inputs
        self.mean = torch.mean(inputs,dim=0)
        self.std = torch.std(inputs,dim=0)
        self.max = torch.max(inputs,dim=0).values
        self.min = torch.min(inputs,dim=0).values
        self.bound = self.max-self.min
    def normalize(self, req_inp):
        return (req_inp-self.mean)/self.std
    def inv_normalize(self, req_inp):
        return (req_inp*self.std+self.mean)
    def uniform_transform(self, req_inp):
        return (req_inp-self.min)/self.bound
    def uniform_inv_transform(self, req_inp):
        return req_inp*self.bound+self.min
    
