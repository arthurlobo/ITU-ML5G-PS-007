#
#
# ===========================================================================
#
#
#
#                            PUBLIC DOMAIN NOTICE
#
#               Naval Surface Warfare Center - Crane Division
#
#
#
#  This software/database is a "United States Government Work" under the
#
#  terms of the United States Copyright Act.  It was written as part of
#
#  the author's official duties as a United States Government employee and
#
#  thus cannot be copyrighted.  This software/database is freely available
#
#  to the public for use. Naval Surface Warfare Center - Crane Division
#
#  (NSWC-CD) and the U.S. Government have not placed any restriction on
#
#  its use or reproduction.
#
#
#
#  Although all reasonable efforts have been taken to ensure the accuracy
#
#  and reliability of the software and data, NSWC-CD and the U.S.
#
#  Government do not and cannot warrant the performance or results that
#
#  may be obtained by using this software or data. NSWC-CD and the U.S.
#
#  Government disclaim all warranties, express or implied, including
#
#  warranties of performance, merchantability or fitness for any particular
#
#  purpose.
#
#
#
#  Please cite the author in any work or product based on this material.
#
#
#
# ===========================================================================
#
# Author: Dr. Arthur Lobo
# Naval Surface Warfare Center, Crane, Indiana
# USA
# Date: October 7, 2021
#

#!/usr/bin/env python
# coding: utf-8

# # Training a Quantized NN for Modulation Classification
# This notebook serves as a starting point for the [Lightning-Fast Modulation Classification with Hardware-Efficient Neural Networks](http://bit.ly/brevitas-radioml-challenge-21) problem statement of the [**ITU AI/ML in 5G Challenge**](https://aiforgood.itu.int/ai-ml-in-5g-challenge/).
# We will show how to create, train, and evaluate an exemplary quantized CNN model to make you familiar with the dataset, task, and provided infrastructure.
# 
# ## Outline
# * [Load the RadioML 2018 Dataset](#load_dataset)
# * [Define the quantized VGG10 Model](#define_model)
#     * [Train the Model from Scratch](#train_model)
#     * [**Alternatively:** Load Pre-Trained Parameters](#load_trained_model)
# * [Evaluate the Accuracy](#evaluate_accuracy)
# * [Evaluate the Inference Cost](#evaluate_inference_cost)

# In[1]:


# Import some general modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn.utils.prune as prune
import sys


# In[2]:


# Select which GPU to use (if available)
gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print("Using GPU %d" % gpu)
else:
    gpu = None
    print("Using CPU only")


# # The RadioML 2018 Dataset <a id='load_dataset'></a>
# This problem statement is based on the popular RadioML 2018.01A dataset provided by DeepSig. It is the latest in a series of modulation classification datasets ([deepsig.ai/datasets](https://www.deepsig.ai/datasets)) and contains samples of 24 digital and analog modulation types under various channel impairments and signal-to-noise ratios (SNRs). For more information on the dataset origins, we refer to the associated paper [Over-the-Air Deep Learning Based Radio Signal Classification](https://arxiv.org/pdf/1712.04578.pdf) by O’Shea, Roy, and Clancy.
# 
# 
# The dataset comes in hdf5 format and exhibits the following structure:
# - 24 modulations
# - 26 SNRs per modulation (-20 dB through +30 dB in steps of 2)
# - 4096 frames per modulation-SNR combination
# - 1024 complex time-series samples per frame
# - Samples as floating point in-phase and quadrature (I/Q) components, resulting in a (1024,2) frame shape
# - 2.555.904 frames in total
#  
# 
# ## Download
# The dataset is available here: **https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz**
# 
# Since access requires a (straightforward) registration, you must download and extract it manually. It measures about 18 GiB in size (20 GiB uncompressed).
# 
# To access it from within this container, you can place it:
# - A) Under the sandbox directory you launched this notebook from, which is mounted under "/workspace/sandbox".
# - B) Anywhere, then set the environment variable `DATASET_DIR` before launching "run_docker.sh" to mount it under "/workspace/dataset".
# 
# You might notice that the dataset comes with a "classes.txt" file containing the alleged modulation labels. However, you should disregard the ordering of these labels due to a known issue ([github.com/radioML/dataset/issues/25](http://github.com/radioML/dataset/issues/25)). This notebook uses the corrected labels throughout.
# 
# In the following, we create the data loader and can inspect some frames to get an idea what the input data looks like.

# In[3]:


# Check if dataset is present
import os.path
dataset_path = "/workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
os.path.isfile(dataset_path)


# In[4]:


# Prepare data loader
from torch.utils.data import Dataset, DataLoader
import h5py

class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24): # all modulations (0 to 23)
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096)) 
                np.random.shuffle(indices_subclass)

                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]

#                train_indices_subclass = indices_subclass[800:1520]
#                test_indices_subclass = indices_subclass[1520:1600]

#                train_indices_subclass = indices_subclass[0:360]
#                test_indices_subclass = indices_subclass[360:400]
                
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)
                test_indices.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

dataset = radioml_18_dataset(dataset_path)


# In[5]:


# Inspect a frame
#mod = 12 # 0 to 23
#snr_idx = 25 # 0 to 25 = -20dB to +30dB
#sample = 123 # 0 to 4095
#-----------------------#
#idx = 26*4096*mod + 4096*snr_idx + sample
#data, mod, snr = dataset.data[idx], dataset.mod[idx], dataset.snr[idx]
#plt.figure(figsize=(12,4))
#plt.plot(data)
#print("Modulation: %s, SNR: %.1f dB, Index: %d" % (dataset.mod_classes[mod], snr, idx))


# # Define the QNN Model <a id='define_model'></a>
# 
# <div>
# <img align="right" width="274" height="418" src="attachment:VGG10_small.png">
# </div>
# 
# As a simple example, we will create a quantized version of the "VGG10" CNN architecture proposed by the dataset authors in [Over-the-Air Deep Learning Based Radio Signal Classification](https://arxiv.org/pdf/1712.04578.pdf).
# 
# Quantizing a sequential pytorch model is straightforward with Brevitas. Relevant `torch.nn` layers are simply replaced by their `brevitas.nn` counterparts, which add customizable input, output, or parameter quantization. Regular Torch layers, especially those that are invariant to quantization (e.g. BatchNorm or MaxPool), can be mixed and matched with Brevitas layers.
# 
# As a baseline, we apply 8-bit quantization to the activations and weights of every layer, except for the final classification output. The input data is quantized to 8 bits with a dedicated quantization layer. Instead of letting Brevitas determine the quantization scale automatically, we set a fixed quantization range (-2.0, 2.0) based on analysis of the whole dataset. Except for two outlier classes (both single-sideband (SSB) modulations), the vast majority of samples (98.3%) at +30 dB fall within this range and will thus not be clipped.
# 
# For more information on Brevitas you can turn to these resources:
# - [GitHub repository](https://github.com/Xilinx/brevitas)
# - [Tutorial notebooks](https://github.com/Xilinx/brevitas/tree/master/notebooks)
# - [Example models](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples)
# - Public discussion in the [Brevitas Gitter channel](https://gitter.im/xilinx-brevitas/community)

# In[6]:


from torch import nn
import brevitas.nn as qnn
from brevitas.quant import IntBias
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.defaults import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat

# Adjustable hyperparameters
input_bits = 8
a_bits = 8
w_bits = 8
filters_conv = 64
filters_dense = 128

# Setting seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


bw = np.zeros(110, dtype=int)

BN_MOMENTUM = 0.001
BN_EPSILON = 1e-3

class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None

def _make_divisible(channels, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_channels = max(min_value, int(channels+divisor/2)//divisor*divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels

class SandGlassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction, bw_idx):
        super(SandGlassBlock, self).__init__()

        self.conv = nn.Sequential(
            # depthwise
#            nn.Conv1d(in_channels, in_channels, 5, 1, 2, groups=in_channels, bias=False),
            qnn.QuantConv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, weight_bit_width=int(bw[bw_idx]), groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels), #, eps=BN_EPSILON, momentum=BN_MOMENTUM),
#            nn.ReLU6(),
            qnn.QuantReLU(bit_width=int(bw[bw_idx+1])),
            # pointwise reduction
#            nn.Conv1d(in_channels, in_channels//reduction, 1, 1, 0, bias=False),
            qnn.QuantConv1d(in_channels, in_channels//reduction, kernel_size=1, stride=1, padding=0, weight_bit_width=int(bw[bw_idx+2]), bias=False),
            nn.BatchNorm1d(in_channels//reduction), #, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            # pointwise expansion
#            nn.Conv1d(in_channels//reduction, out_channels, 1, 1, 0, bias=False),
            qnn.QuantConv1d(in_channels//reduction, out_channels, kernel_size=1, stride=1, padding=0, weight_bit_width=int(bw[bw_idx+3]), bias=False),
            nn.BatchNorm1d(out_channels), #, eps=BN_EPSILON, momentum=BN_MOMENTUM),
#            nn.ReLU6(),
            qnn.QuantReLU(bit_width=int(bw[bw_idx+4])),
            # depthwise
#            nn.Conv1d(out_channels, out_channels, 5, stride, 2, groups=out_channels, bias=False),
            qnn.QuantConv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, weight_bit_width=int(bw[bw_idx+5]), groups=out_channels, bias=False),
            nn.BatchNorm1d(out_channels) #, eps=BN_EPSILON, momentum=BN_MOMENTUM)
        )

        self.residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        if self.residual:
            return self.conv(x) + x
        else:
            return self.conv(x)

class MobileNeXt(nn.Module):
    config = [

        # channels (c), stride (s), reduction (t = expansion factor), blocks (n=repeating number)
         [16,   1, 1, 1],
         [24,   2, 6, 2],
         [32,   2, 6, 3],
         [64,   2, 6, 4],
         [96,   1, 6, 3],
         [160,  2, 6, 3],
         [320,  1, 6, 1],
    ]

    def __init__(self, input_size=1024, num_classes=24, width_mult=1., i1=8):
        super(MobileNeXt, self).__init__()

        self.qnn_input = qnn.QuantIdentity(bit_width=i1, return_quant_tensor=True)
        stem_channels = 32 
        stem_channels = _make_divisible(int(stem_channels*width_mult), 8)
        self.conv_stem = nn.Sequential(
#            nn.Conv1d(2, stem_channels, 3, 2, 1, bias=False),
            qnn.QuantConv1d(2, stem_channels, kernel_size=3, stride=2, padding=1, weight_bit_width=int(bw[0]), bias=False),
            nn.BatchNorm1d(stem_channels), #, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            qnn.QuantReLU(bit_width=int(bw[1]))
#            nn.ReLU6()
        )

        blocks = []
        in_channels = stem_channels
        bw_idx=2
        for c, s, r, b in self.config:
            out_channels = _make_divisible(int(c*width_mult), 8)
#            print(' ', out_channels)
            for i in range(b):
                stride = s if i == 0 else 1
                blocks.append(SandGlassBlock(in_channels, out_channels, stride, r, bw_idx))
                in_channels = out_channels
                bw_idx += 6

        self.out_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
#        self.final_pool = qnn.QuantAvgPool2d(kernel_size=7, stride=1, bit_width=8) #int(bw[bw_idx]))
#        self.classifier = nn.Linear(out_channels, num_classes)
        self.classifier = qnn.QuantLinear(out_channels, 24, weight_bit_width=int(bw[bw_idx]), bias=True) #, bias_quant=IntBias)

    def forward(self, x):
        x = self.qnn_input(x)
        x = self.conv_stem(x)
        x = self.blocks(x)     # [1024, 64, 512]
        x = self.avg_pool(x)   # [1024, 64, 1]
#        x = x.view(-1, x.size(1))
        x = x.view(x.shape[0], -1)
#        x = x.flatten(1) 
#        print(self.out_channels)  # 64
        y = self.classifier(x)

        return y


class InputQuantizer(Int8ActPerTensorFloatMinMaxInit):
    bit_width = input_bits
    min_val = -2.0
    max_val = 2.0
    scaling_impl_type = ScalingImplType.CONST # Fix the quantization range to [min_val, max_val]

class Net(nn.Module):
    def __init__(self, l1=64, l2=128, w1=8, a1=8, w2=8, a2=8, w3=8, a3=8, w4=8, a4=8, w5=8, a5=8, w6=8, a6=8, w7=8, a7=8, w8=8, a8=8, w9=8, a9=8, w10=8, i1=8):
        super(Net, self).__init__()

#        self.qnn_input = qnn.QuantHardTanh(act_quant=InputQuantizer)

        self.qnn_input = qnn.QuantIdentity(bit_width=i1, return_quant_tensor=True)


        self.conv1 = qnn.QuantConv1d(2, filters_conv, 3, padding=1, weight_bit_width=w1, bias=False)
        self.bn1   = nn.BatchNorm1d(filters_conv)
        self.relu1 = qnn.QuantReLU(bit_width=a1) 
        self.pool1 = nn.MaxPool1d(2) 

        self.conv2 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w2, bias=False)
        self.bn2   = nn.BatchNorm1d(filters_conv)
        self.relu2 = qnn.QuantReLU(bit_width=a2) 
        self.pool2 = nn.MaxPool1d(2) 


        self.conv3 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w3, bias=False)
        self.bn3   = nn.BatchNorm1d(filters_conv)
        self.relu3 = qnn.QuantReLU(bit_width=a3) 
        self.pool3 = nn.MaxPool1d(2) 


        self.conv4 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w4, bias=False)
        self.bn4   = nn.BatchNorm1d(filters_conv)
        self.relu4 = qnn.QuantReLU(bit_width=a4) 
        self.pool4 = nn.MaxPool1d(2) 


        self.conv5 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w5, bias=False)
        self.bn5   = nn.BatchNorm1d(filters_conv)
        self.relu5 = qnn.QuantReLU(bit_width=a5) 
        self.pool5 = nn.MaxPool1d(2) 


        self.conv6 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w6, bias=False)
        self.bn6   = nn.BatchNorm1d(filters_conv)
        self.relu6 = qnn.QuantReLU(bit_width=a6) 
        self.pool6 = nn.MaxPool1d(2) 


        self.conv7 = qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w7, bias=False)
        self.bn7   = nn.BatchNorm1d(filters_conv)
        self.relu7 = qnn.QuantReLU(bit_width=a7) 
        self.pool7 = nn.MaxPool1d(2) 


        self.flatten = nn.Flatten()


        self.fc1   = qnn.QuantLinear(filters_conv*8, l1, weight_bit_width=w8, bias=False)
        self.bn8   = nn.BatchNorm1d(l1)
        self.relu8 = qnn.QuantReLU(bit_width=a8)


        self.fc2   = qnn.QuantLinear(l1, l2, weight_bit_width=w9, bias=False)
        self.bn9   = nn.BatchNorm1d(l2)
        self.relu9 = qnn.QuantReLU(bit_width=a9, return_quant_tensor=True)


        self.fc3   = qnn.QuantLinear(l2, 24, weight_bit_width=w10, bias=True, bias_quant=IntBias)


    def forward(self, x):

        x = self.qnn_input(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))

        x = self.flatten(x)

        x = self.relu8(self.bn8(self.fc1(x)))
        x = self.relu9(self.bn9(self.fc2(x)))
        x = self.fc3(x)
        return x


model1 = nn.Sequential(
    # Input quantization layer
    qnn.QuantHardTanh(act_quant=InputQuantizer),

    qnn.QuantConv1d(2, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits,bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),

    qnn.QuantConv1d(filters_conv, filters_conv, 3, padding=1, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_conv),
    qnn.QuantReLU(bit_width=a_bits),
    nn.MaxPool1d(2),
    
    nn.Flatten(),

    qnn.QuantLinear(filters_conv*8, filters_dense, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_dense),
    qnn.QuantReLU(bit_width=a_bits),

    qnn.QuantLinear(filters_dense, filters_dense, weight_bit_width=w_bits, bias=False),
    nn.BatchNorm1d(filters_dense),
    qnn.QuantReLU(bit_width=a_bits, return_quant_tensor=True),

    qnn.QuantLinear(filters_dense, 24, weight_bit_width=w_bits, bias=True, bias_quant=IntBias),
)


# # Train the QNN from Scratch <a id='train_model'></a>
# <span style="color:red">Even with GPU acceleration, training will take multiple minutes per epoch!<br>You can skip this section and load a pre-trained model instead: [Load Pre-Trained Parameters](#load_trained_model)</span>
# 
# First, we define basic train and test functions, which will be called for each training epoch. Training itself follows the usual Pytorch procedures, while Brevitas handles all quantization-specifics automatically in the background.

# In[7]:


from sklearn.metrics import accuracy_score


def display_loss_plot(losses, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Now we can start the training loop for a number of epochs.
# 
# If you run into VRAM limitations of your system, it might help to decrease the `batch_size` and initial learning rate accordingly. To keep this notebook's resource footprint small, we do not pre-load the whole dataset into DRAM. You should adjust your own training code to take advantage of multiprocessing and available memory for maximum performance.

# In[8]:


batch_size = 1024
num_epochs = 100 


data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from finn.util.inference_cost import inference_cost
import json

export_onnx_path = "models/model_export.onnx"
final_onnx_path = "models/model_final.onnx"
cost_dict_path = "models/model_cost.json"

bops_baseline = 807699904
w_bits_baseline = 1244936


def train_modclass():

    print(lr, wd, i1, bw[0:105])

    model = MobileNeXt(1024, 24, 1., i1)

    savefile = "MobileNeXt_quantized_8_bit.pth"
    saved_state = torch.load(savefile, map_location=torch.device("cpu"))
    model.load_state_dict(saved_state)
    print('loaded .pth')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    if gpu is not None:
      model = model.cuda()

    prune_frac = 0.941

    parameters_to_prune = (
      (model.conv_stem[0], 'weight'),

      (model.blocks[0].conv[0], 'weight'),
      (model.blocks[0].conv[3], 'weight'),
      (model.blocks[0].conv[5], 'weight'),
      (model.blocks[0].conv[8], 'weight'),

      (model.blocks[1].conv[0], 'weight'),
      (model.blocks[1].conv[3], 'weight'),
      (model.blocks[1].conv[5], 'weight'),
      (model.blocks[1].conv[8], 'weight'),

      (model.blocks[2].conv[0], 'weight'),
      (model.blocks[2].conv[3], 'weight'),
      (model.blocks[2].conv[5], 'weight'),
      (model.blocks[2].conv[8], 'weight'),

      (model.blocks[3].conv[0], 'weight'),
      (model.blocks[3].conv[3], 'weight'),
      (model.blocks[3].conv[5], 'weight'),
      (model.blocks[3].conv[8], 'weight'),

      (model.blocks[4].conv[0], 'weight'),
      (model.blocks[4].conv[3], 'weight'),
      (model.blocks[4].conv[5], 'weight'),
      (model.blocks[4].conv[8], 'weight'),

      (model.blocks[5].conv[0], 'weight'),
      (model.blocks[5].conv[3], 'weight'),
      (model.blocks[5].conv[5], 'weight'),
      (model.blocks[5].conv[8], 'weight'),

      (model.blocks[6].conv[0], 'weight'),
      (model.blocks[6].conv[3], 'weight'),
      (model.blocks[6].conv[5], 'weight'),
      (model.blocks[6].conv[8], 'weight'),

      (model.blocks[7].conv[0], 'weight'),
      (model.blocks[7].conv[3], 'weight'),
      (model.blocks[7].conv[5], 'weight'),
      (model.blocks[7].conv[8], 'weight'),

      (model.blocks[8].conv[0], 'weight'),
      (model.blocks[8].conv[3], 'weight'),
      (model.blocks[8].conv[5], 'weight'),
      (model.blocks[8].conv[8], 'weight'),

      (model.blocks[9].conv[0], 'weight'),
      (model.blocks[9].conv[3], 'weight'),
      (model.blocks[9].conv[5], 'weight'),
      (model.blocks[9].conv[8], 'weight'),

      (model.blocks[10].conv[0], 'weight'),
      (model.blocks[10].conv[3], 'weight'),
      (model.blocks[10].conv[5], 'weight'),
      (model.blocks[10].conv[8], 'weight'),

      (model.blocks[11].conv[0], 'weight'),
      (model.blocks[11].conv[3], 'weight'),
      (model.blocks[11].conv[5], 'weight'),
      (model.blocks[11].conv[8], 'weight'),

      (model.blocks[12].conv[0], 'weight'),
      (model.blocks[12].conv[3], 'weight'),
      (model.blocks[12].conv[5], 'weight'),
      (model.blocks[12].conv[8], 'weight'),

      (model.blocks[13].conv[0], 'weight'),
      (model.blocks[13].conv[3], 'weight'),
      (model.blocks[13].conv[5], 'weight'),
      (model.blocks[13].conv[8], 'weight'),

      (model.blocks[14].conv[0], 'weight'),
      (model.blocks[14].conv[3], 'weight'),
      (model.blocks[14].conv[5], 'weight'),
      (model.blocks[14].conv[8], 'weight'),

      (model.blocks[15].conv[0], 'weight'),
      (model.blocks[15].conv[3], 'weight'),
      (model.blocks[15].conv[5], 'weight'),
      (model.blocks[15].conv[8], 'weight'),

      (model.blocks[16].conv[0], 'weight'),
      (model.blocks[16].conv[3], 'weight'),
      (model.blocks[16].conv[5], 'weight'),
      (model.blocks[16].conv[8], 'weight'),

      (model.classifier, 'weight'),
    )

#   Pruning is not done 
#
#    prune.global_unstructured(
#      parameters_to_prune,
#      pruning_method=prune.L1Unstructured,
#      amount=prune_frac,
#    )

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if gpu is not None:
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-8) #, weight_decay=wd)

#    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, steps_per_epoch=len(data_loader_train), epochs=num_epochs)

    running_loss = []
    running_test_acc = []
    test_acc_max = 0.56 #0.560464
    test_IC_min = 0.054512

    l1_lambda = wd 
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
#        loss_epoch = train(model, data_loader_train, optimizer, criterion)

        model = model.cuda()
        losses = []

        # ensure model is in training mode
        model.train()
        i = 0
        for (inputs, target, snr) in tqdm(data_loader_train, desc="Batches", leave=False):
            print(i, '\r', end = '', flush=True)
            if gpu is not None:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward pass
            output = model(inputs)
            loss = criterion(output, target)

            # L1 regularization
            L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)

            # Alternative method
            # L1_reg = sum(p.abs().sum() for p in model.parameters())

            loss = loss + l1_lambda * L1_reg


            # backward pass + run optimizer to update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # keep track of loss value
            losses.append(loss.cpu().detach().numpy())
#            lr_scheduler.step()
            i += 1
            
        loss_epoch = losses
#        print('lr = ', lr_scheduler.get_last_lr())


#        test_acc = test(model, data_loader_test)


        # ensure model is in eval mode
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for (inputs, target, snr) in data_loader_test:
                if gpu is not None:
                    inputs = inputs.cuda()
                    target = target.cuda()
                output = model(inputs)
                pred = output.argmax(dim=1, keepdim=True)
                y_true.extend(target.tolist())
                y_pred.extend(pred.reshape(-1).tolist())

        test_acc = accuracy_score(y_true, y_pred)



        BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path);
        inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
                  preprocess=True, discount_sparsity=True)


        with open(cost_dict_path, 'r') as f:
          inference_cost_dict = json.load(f)

        bops = int(inference_cost_dict["total_bops"])
        w_bits = int(inference_cost_dict["total_mem_w_bits"])


        score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)

        if (test_acc > test_acc_max) and (score < test_IC_min):
#          test_acc_max = test_acc
          test_IC_min = score
          torch.save(model.state_dict(), "MobileNeXt_quantized_8_bit.pth")

        print("Epoch %d: Training loss = %f, test accuracy = %f, inference cost = %f" % (epoch, np.mean(loss_epoch), test_acc, score))
#        running_loss.append(loss_epoch)
#        running_test_acc.append(test_acc)
#        lr_scheduler.step()



    return score, test_acc 



#parms = {'wd': 4.264073558832227e-05, 'lr': 0.5877787668535266, 'w0': 8, 'w1': 8, 'w2': 3, 'w3': 4, 'w4': 5, 'w5': 5, 'w6': 5, 'w7': 5, 'w8': 5, 'w9': 8, 'w10': 2, 'w11': 4, 'w12': 6, 'w13': 7, 'w14': 5, 'w15': 2, 'w16': 8, 'w17': 5, 'w18': 6, 'w19': 4, 'w20': 4, 'w21': 7, 'w22': 4, 'w23': 5, 'w24': 7, 'w25': 7, 'w26': 7, 'w27': 8, 'w28': 7, 'w29': 2, 'w30': 8, 'w31': 2, 'w32': 3, 'w33': 7, 'w34': 4, 'w35': 4, 'w36': 2, 'w37': 2, 'w38': 2, 'w39': 6, 'w40': 4, 'w41': 6, 'w42': 4, 'w43': 5, 'w44': 2, 'w45': 5, 'w46': 7, 'w47': 8, 'w48': 7, 'w49': 6, 'w50': 7, 'w51': 8, 'w52': 7, 'w53': 2, 'w54': 4, 'w55': 8, 'w56': 8, 'w57': 5, 'w58': 2, 'w59': 7, 'w60': 2, 'w61': 7, 'w62': 8, 'w63': 3, 'w64': 3, 'w65': 3, 'w66': 3, 'w67': 8, 'w68': 5, 'w69': 7, 'w70': 2, 'w71': 7, 'w72': 7, 'w73': 2, 'w74': 3, 'w75': 6, 'w76': 5, 'w77': 5, 'w78': 2, 'w79': 2, 'w80': 7, 'w81': 2, 'w82': 4, 'w83': 3, 'w84': 7, 'w85': 3, 'w86': 2, 'w87': 4, 'w88': 7, 'w89': 2, 'w90': 5, 'w91': 2, 'w92': 2, 'w93': 3, 'w94': 3, 'w95': 8, 'w96': 8, 'w97': 8, 'w98': 4, 'w99': 8, 'w100': 2, 'w101': 3, 'w102': 5, 'w103': 7, 'w104': 6, 'i1': 6}

#test accuracy = 0.552564, inference cost = 0.146154
#parms = {'wd': 2.7997931408569504e-06, 'lr': 0.1665785930572429, 'w0': 8, 'w1': 6, 'w2': 6, 'w3': 5, 'w4': 2, 'w5': 3, 'w6': 7, 'w7': 7, 'w8': 5, 'w9': 6, 'w10': 3, 'w11': 8, 'w12': 8, 'w13': 7, 'w14': 3, 'w15': 3, 'w16': 6, 'w17': 7, 'w18': 2, 'w19': 3, 'w20': 5, 'w21': 2, 'w22': 5, 'w23': 8, 'w24': 6, 'w25': 8, 'w26': 2, 'w27': 5, 'w28': 6, 'w29': 6, 'w30': 5, 'w31': 7, 'w32': 5, 'w33': 6, 'w34': 8, 'w35': 2, 'w36': 4, 'w37': 7, 'w38': 8, 'w39': 6, 'w40': 7, 'w41': 3, 'w42': 3, 'w43': 7, 'w44': 6, 'w45': 3, 'w46': 3, 'w47': 8, 'w48': 5, 'w49': 8, 'w50': 2, 'w51': 8, 'w52': 4, 'w53': 3, 'w54': 5, 'w55': 5, 'w56': 6, 'w57': 4, 'w58': 2, 'w59': 8, 'w60': 6, 'w61': 4, 'w62': 7, 'w63': 7, 'w64': 6, 'w65': 4, 'w66': 7, 'w67': 3, 'w68': 8, 'w69': 8, 'w70': 8, 'w71': 4, 'w72': 2, 'w73': 2, 'w74': 2, 'w75': 4, 'w76': 4, 'w77': 5, 'w78': 7, 'w79': 2, 'w80': 5, 'w81': 7, 'w82': 7, 'w83': 5, 'w84': 7, 'w85': 6, 'w86': 3, 'w87': 3, 'w88': 4, 'w89': 5, 'w90': 4, 'w91': 2, 'w92': 4, 'w93': 3, 'w94': 7, 'w95': 5, 'w96': 3, 'w97': 4, 'w98': 8, 'w99': 6, 'w100': 3, 'w101': 2, 'w102': 5, 'w103': 8, 'w104': 8, 'i1': 7}

#test accuracy = 0.536799, inference cost = 0.105566
parms = {'wd': 6.266221520264738e-06, 'lr': 0.2990467901901732, 'w0': 8, 'w1': 7, 'w2': 8, 'w3': 7, 'w4': 3, 'w5': 4, 'w6': 8, 'w7': 2, 'w8': 3, 'w9': 2, 'w10': 4, 'w11': 5, 'w12': 5, 'w13': 5, 'w14': 5, 'w15': 8, 'w16': 2, 'w17': 5, 'w18': 7, 'w19': 2, 'w20': 8, 'w21': 6, 'w22': 3, 'w23': 5, 'w24': 2, 'w25': 4, 'w26': 8, 'w27': 2, 'w28': 7, 'w29': 8, 'w30': 7, 'w31': 4, 'w32': 8, 'w33': 3, 'w34': 8, 'w35': 4, 'w36': 7, 'w37': 6, 'w38': 3, 'w39': 7, 'w40': 5, 'w41': 3, 'w42': 8, 'w43': 3, 'w44': 2, 'w45': 6, 'w46': 5, 'w47': 2, 'w48': 2, 'w49': 6, 'w50': 3, 'w51': 5, 'w52': 5, 'w53': 5, 'w54': 5, 'w55': 8, 'w56': 4, 'w57': 2, 'w58': 4, 'w59': 4, 'w60': 3, 'w61': 3, 'w62': 5, 'w63': 5, 'w64': 2, 'w65': 3, 'w66': 8, 'w67': 4, 'w68': 8, 'w69': 3, 'w70': 7, 'w71': 2, 'w72': 8, 'w73': 3, 'w74': 5, 'w75': 4, 'w76': 7, 'w77': 4, 'w78': 8, 'w79': 7, 'w80': 7, 'w81': 3, 'w82': 5, 'w83': 5, 'w84': 8, 'w85': 6, 'w86': 4, 'w87': 7, 'w88': 6, 'w89': 3, 'w90': 6, 'w91': 2, 'w92': 7, 'w93': 7, 'w94': 2, 'w95': 7, 'w96': 4, 'w97': 4, 'w98': 7, 'w99': 8, 'w100': 8, 'w101': 2, 'w102': 3, 'w103': 5, 'w104': 7, 'i1': 7}

bw = np.zeros(106, dtype=int)
i=0
for k, v in parms.items():
#  print(k, v)
  if (i>=2):
    bw[i-2] = 8 #int(v)
  i+=1

wd = 6.266221520264738e-06
lr = 0.2990467901901732
i1 = 8 #int(bw[105])

train = int(sys.argv[1])

if train == 1:
  train_modclass()


# In[13]:


# Plot training loss over epochs
# loss_per_epoch = [np.mean(loss_per_epoch) for loss_per_epoch in running_loss]
# display_loss_plot(loss_per_epoch)


# In[14]:


# Plot test accuracy over epochs
# acc_per_epoch = [np.mean(acc_per_epoch) for acc_per_epoch in running_test_acc]
# display_loss_plot(acc_per_epoch, title="Test accuracy", ylabel="Accuracy [%]")


# In[16]:


# Save the trained parameters to disk
# torch.save(model.state_dict(), "model_trained.pth")


# # Load a Trained Model <a id='load_trained_model'></a>
# Alternatively, you can load the provided pre-trained model.
# It was trained for 20 epochs and reaches an overall accuracy of 59.5%.

# In[8]:


# Load trained parameters
model = MobileNeXt(1024, 24, 1., i1)
savefile = "./MobileNeXt_quantized_8_bit.pth"
saved_state = torch.load(savefile, map_location=torch.device("cpu"))
model.load_state_dict(saved_state)
if gpu is not None:
    model = model.cuda()


# # Evaluate the Accuracy <a id='evaluate_accuracy'></a>
# The following cells visualize the test accuracy across different modulations and signal-to-noise ratios. Submissions for this problem statement must reach an overall accuracy of at least **56.0%**, so this should give you an idea what makes up this figure.

# In[9]:


# Set up a fresh test data loader
batch_size = 1024
dataset = radioml_18_dataset(dataset_path)
data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)


# In[10]:


# Run inference on validation data
y_exp = np.empty((0))
y_snr = np.empty((0))
y_pred = np.empty((0,len(dataset.mod_classes)))
model.eval()
with torch.no_grad():
    for data in tqdm(data_loader_test, desc="Batches"):
        inputs, target, snr = data
        if gpu is not None:
            inputs = inputs.cuda()
        output = model(inputs)
        y_pred = np.concatenate((y_pred,output.cpu()))
        y_exp = np.concatenate((y_exp,target))
        y_snr = np.concatenate((y_snr,snr))


# In[11]:


# Plot overall confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
for i in range(len(y_exp)):
    j = int(y_exp[i])
    k = int(np.argmax(y_pred[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(dataset.mod_classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

plt.figure(figsize=(12,8))
plot_confusion_matrix(confnorm, labels=dataset.mod_classes)

cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
print("Overall Accuracy across all SNRs: %f"%(cor / (cor+ncor)))


# In[12]:


# Plot confusion matrices at 4 different SNRs
snr_to_plot = [-20,-4,+4,+30]
plt.figure(figsize=(16,10))
acc = []
for snr in dataset.snr_classes:
    # extract classes @ SNR
    indices_snr = (y_snr == snr).nonzero()
    y_exp_i = y_exp[indices_snr]
    y_pred_i = y_pred[indices_snr]
 
    conf = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
    confnorm = np.zeros([len(dataset.mod_classes),len(dataset.mod_classes)])
    for i in range(len(y_exp_i)):
        j = int(y_exp_i[i])
        k = int(np.argmax(y_pred_i[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(dataset.mod_classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
 
    if snr in snr_to_plot:
        plot, = np.where(snr_to_plot == snr)[0]
        plt.subplot(221+plot)
        plot_confusion_matrix(confnorm, labels=dataset.mod_classes, title="Confusion Matrix @ %d dB"%(snr))
 
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc.append(cor/(cor+ncor))


# In[13]:


# Plot accuracy over SNR
plt.figure(figsize=(10,6))
plt.plot(dataset.snr_classes, acc, marker='o')
plt.xlabel("SNR [dB]")
plt.xlim([-20, 30])
plt.ylabel("Classification Accuracy")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Classification Accuracy over SNR")
plt.grid()
plt.title("Classification Accuracy over SNR");

print("Accuracy @ highest SNR (+30 dB): %f"%(acc[-1]))
print("Accuracy overall: %f"%(np.mean(acc)))


# In[14]:


# Plot accuracy per modulation
accs = []
for mod in range(24):
    accs.append([])
    for snr in dataset.snr_classes:
        indices = ((y_exp == mod) & (y_snr == snr)).nonzero()
        y_exp_i = y_exp[indices]
        y_pred_i = y_pred[indices]
        cor = np.count_nonzero(y_exp_i == np.argmax(y_pred_i, axis=1))
        accs[mod].append(cor/len(y_exp_i))
        
# Plot accuracy-over-SNR curve
plt.figure(figsize=(12,8))
for mod in range(24):
    if accs[mod][25] < 0.95 or accs[mod][0] > 0.1:
        color = None
    else:
        color = "black"
    plt.plot(dataset.snr_classes, accs[mod], label=str(mod) + ": " + dataset.mod_classes[mod], color=color)
plt.xlabel("SNR [dB]")
plt.xlim([-20, 30])
plt.ylabel("Classification Accuracy")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Accuracy breakdown")
plt.grid()
plt.legend();


# # Evaluate the Inference Cost <a id='evaluate_inference_cost'></a>
# 
# First, we have to export the model to Brevita's quantized variant of the ONNX interchange format. **All submissions must correctly pass through this export flow and provide the resulting .onnx file**. Any `TracerWarning` can be safely ignored.

# In[15]:


from brevitas.export.onnx.generic.manager import BrevitasONNXManager

export_onnx_path = "models/model_export.onnx"
final_onnx_path = "models/model_final.onnx"
cost_dict_path = "models/model_cost.json"

BrevitasONNXManager.export(model.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path);


# Now we use our analysis tool, which is part of [finn-base](https://github.com/Xilinx/finn-base), to determine the inference cost. It reports the number of output activation variables (`mem_o`), weight parameters (`mem_w`), and multiply-accumulate operations (`op_mac`) for each data type. These are used to calculate the total number of activation bits, weight bits, and bit-operations (BOPS).
# 
# If the report shows any unsupported operations, for instance because you implemented custom layers, you should check with the rules on the problem statement [website](http://bit.ly/brevitas-radioml-challenge-21) and consider to contact the organizers.

# In[16]:


from finn.util.inference_cost import inference_cost
import json

inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path,
               preprocess=True, discount_sparsity=True)


# The call to `ìnference_cost()` cleans up the model by inferring shapes and datatypes, folding constants, etc. We visualize the pre-processed ONNX model using [Netron](https://netron.app/).

# In[25]:


#import os
#import netron
#from IPython.display import IFrame

#def showInNetron(model_filename):
#    localhost_url = os.getenv("LOCALHOST_URL")
#    netron_port = os.getenv("NETRON_PORT")
#    netron.start(model_filename, address=("0.0.0.0", int(netron_port)))
#    return IFrame(src="http://%s:%s/" % (localhost_url, netron_port), width="100%", height=400)

#showInNetron(final_onnx_path)


plt.show()


# Finally, we compute the inference cost score, normalized to the baseline 8-bit VGG10 defined in this notebook. **Submissions will be judged based on this score.**

# In[17]:


with open(cost_dict_path, 'r') as f:
    inference_cost_dict = json.load(f)

bops = int(inference_cost_dict["total_bops"])
w_bits = int(inference_cost_dict["total_mem_w_bits"])

bops_baseline = 807699904
w_bits_baseline = 1244936

score = 0.5*(bops/bops_baseline) + 0.5*(w_bits/w_bits_baseline)
print("Normalized inference cost score: %f" % score)


# In[ ]:




