import torch

class PreprocessLayer(torch.nn.Module):
    """
    Preprocess layer for the input data , csv data ->  hash,bucket,  
    """
    def __init__(self):
        super(preprocess, self).__init__()

