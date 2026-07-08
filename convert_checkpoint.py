import numpy as np
import torch
import sys

# Gets trues and falses for whether each weight is positive or negative
# np.sign gives +1, -1, 0 (like torch.sign used in the binarization() method)
def ones_minus_ones(x):
    ones = (np.sign(x) == 1.0)
    minus_ones = (np.sign(x) == -1.0)
    return ones, minus_ones

# Interleave them
def interleave_nums(ones, minus_ones):
    interleaved = np.empty((len(ones) + len(minus_ones)), dtype=bool)
    interleaved[0::2] = ones
    interleaved[1::2] = minus_ones
    return interleaved

checkpoint = torch.load("runs/WideResNet-28-10/model_best.pth.tar")

L1 = checkpoint["state_dict"]["1.weight"].detach().numpy().transpose()
L2 = checkpoint["state_dict"]["3.weight"].detach().numpy().transpose()

print(f"input->hidden shape {L1.shape}") 
print(f"hidden->output shape {L2.shape}") 

# **YOUSSEF** because we operate on 4 weights at a time, we need to make sure the number of columns in the weight matrix are a multiple of 4
if (L1.shape[1] % 4) != 0:
    L1 = np.pad(L1, ((0, 0), (0, 4 - (L1.shape[1] % 4))))
    print(f"padded input->hidden shape {L1.shape}")
if (L2.shape[1] % 4) != 0:
    L2 = np.pad(L2, ((0, 0), (0, 4 - (L2.shape[1] % 4))))
    print(f"padded hidden->output shape {L2.shape}")

ones_1, minus_1 = ones_minus_ones(L1.flatten()) 
ones_2, minus_2 = ones_minus_ones(L2.flatten())
L1_interleaved = interleave_nums(ones_1, minus_1)
L2_interleaved = interleave_nums(ones_2, minus_2)

# Pack these into raw bytes - I would copy-paste them straight into an array of uint8_t in your C code
Eight_bit_weights_Layer1 = np.packbits(L1_interleaved, bitorder="little")
Eight_bit_weights_Layer2 = np.packbits(L2_interleaved, bitorder="little")

print(np.array2string(Eight_bit_weights_Layer1, max_line_width=80, separator=",", threshold=sys.maxsize))
print(np.array2string(Eight_bit_weights_Layer2, max_line_width=80, separator=",", threshold=sys.maxsize))
