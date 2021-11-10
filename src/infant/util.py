import torch as T

def prefer_gpu():
    return 'cuda:0' if T.cuda.is_available() else 'cpu'