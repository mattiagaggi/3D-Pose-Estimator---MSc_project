import torch
import numpy as np
from utils.trans_numpy_torch import create_zero_float_tensor, numpy_to_tensor
#can't even be install without cuda
if torch.cuda.is_available():
    import torch_batch_svd
else:
    from torch import svd
from torch import det



def tiled_identity(length):
    ident = np.eye(3)
    iden_batch = np.tile(ident, (length, 1, 1))
    return numpy_to_tensor(iden_batch)



def batch_svd(M):
    if torch.cuda.is_available():
        U, S, VT = torch_batch_svd(M)
    else:
        U, S, VT = [], [], []
        for i in range(M.size()[0]):
            R = svd(M[i])
            U.append(R[0])
            S.append(R[1])
            VT.append(R[2])
        U = torch.stack(U,dim=0)
        S= torch.stack(S, dim=0)
        VT = torch.stack(VT, dim=0)
    return U, S, VT

def determinant(M):
    batch_size=M.size()[0]
    D = create_zero_float_tensor([batch_size])
    for i in range(M.size()[0]):
        D[i]= det(M[i])
    return D
