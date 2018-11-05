import torch
import numpy as np
import math

# ----------------------------------------------------------------------------
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
else:
    device = torch.device("cpu")
    print(device)
# ----------------------------------------------------------------------------
Batch_size = 64  # Batch size
Q = 1000  # Input size
S = 100  # Number of neurons
a = 10  # Network output size
# ----------------------------------------------------------------------------
a0 = torch.randn(Batch_size, Q, device=device, dtype=dtype)
t = torch.randn(Batch_size, a, device=device, dtype=dtype)
# ----------------------------------------------------------------------------
w1 = torch.randn(Q, S, device=device, dtype=dtype)
w2 = torch.randn(S, a, device=device, dtype=dtype)
learning_rate = 1e-6
# ----------------------------------------------------------------------------
for index in range(10):
    n1 = (w1 * a0[index])
    a1 = n1 if n1 > 0 else 0
    n2 = np.mm(w2.t, a1)
    a2 = n2
    loss = (a2 - t).pow(2).sum()
    print(index, loss)

    #     h = p.mm(w1)  #### Matmul
    #     h_relu = h.clamp(min=0)  ### Clamps everything to min of
    #     a_net = h_relu.mm(w2) #Purelin output

    #     loss = (a_net - t).pow(2).sum()
    #     print(index, loss)

    grad_y_pred = 2.0 * (a2 - t)
    grad_w2 = a2.t().m, (grad_y_pred)  ## .t() flips a 2D array
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = p.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2