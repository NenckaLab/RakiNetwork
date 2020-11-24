#!/usr/bin/env python3
# coding: utf-8

import h5py as h5
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import matplotlib.pyplot as plt
import time
import argparse
import os
import pickle
import subprocess

time.sleep(10)

#Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('-sd','--seed',type=int,help='Random generator seed value, default 42',default=42)
parser.add_argument('-dd','--dataDir',type=str,help="Absolute path to input files",\
                    default='/rcc/stor1/depts/radiology/nencka_lab/EpiSMSAI')
parser.add_argument('-cf','--calFile',type=str,help='Filename of calibration data, default unaliasedCalkspace.h5',\
                    default='unaliasedCalkspace.h5')
parser.add_argument('-mf','--muxFile',type=str,help='Filename of alised data, default kspacemux.h5',\
                    default='kspacemux.h5')
parser.add_argument('-p','--packet',type=int,help='Packet to unalias, default 0',default=0)
parser.add_argument('-s','--slice',type=int,help='Slice to unalias, default 4',default=4)
parser.add_argument('-c','--coil',type=int,help='Coil to unalias, default 16',default=16)
parser.add_argument('-r','--smsR',type=int,help='SMS acceleration factor, default 8',default=8)
parser.add_argument('-ba','--batch',type=int,help='Batch size for training, default=48',default=48)
parser.add_argument('-e','--epoch',type=int,help='Epochs for training, default=200',default=200)
parser.add_argument('-lr','--learningRate',type=float,help='Learning rate, default=1e-4',default=1e-4)
parser.add_argument('-nl','--networkLayers',type=int,help='Number of layers in CNN, default=3',default=3)
parser.add_argument('-ks','--kernelSize',type=int,help='Size of convolution kernel, default=7',default=7)
parser.add_argument('-nf','--numFilters',type=int,help='Number of filters to use in most layers, default=64',default=64)
parser.add_argument('-mc','--middleChannels',type=int,help='Number of middle channels in RAKI network, default=1024',\
                    default=1024)
parser.add_argument('-do','--dropOut',type=float,help='Fraction of observations to dropout, default=0.0',default=0.0)
parser.add_argument('-bn','--batchNorm',action="store_true",help='Flag to perform batch normalization, default=False',\
        default=False)
parser.add_argument('-g','--groups',type=int,help='Number of groups to use in CNN, default=1',default=1)
parser.add_argument('-bi','--bias',action='store_true',help='Flag to include bias elements in CNN, default=False',default=False)
parser.add_argument('-rv','--reduceValid',action='store_true',
                    help='Flag to reduce the validation set by taking 1/20 of all values, default=True',default=False)
parser.add_argument('-cs','--caipiShift',type=int,help='FOV shift of CAIPI data, default=3',default=3)
parser.add_argument('-epDur','--epochsDefByDuration',action='store_true',\
        help='Flag to use ep option as duration, in seconds, rather than number of epochs, default=False',\
        default=False)
parser.add_argument('-ss','--splitSlice',action='store_true',help='Flag to include split-slice training, default=True',default=False)
parser.add_argument('-std','--standardTrain',action='store_true',help='Flag to include standard training, default=False',default=False)

#Do the argument parsing
args = parser.parse_args()
print('Running with arguments:')
print(args)
#Manage the randomness
seed=args.seed
if seed!=42:
    print('Random seed has been changed to %i'%(seed))
np.random.seed(seed)
torch.manual_seed(seed)

#Specific to THIS DATA (to be replaced with arguments)
dataDir = args.dataDir
calH5FN = args.calFile #'unaliasedCalkspace.h5'#'ex4302_se11_01-03-2016_104856Cal.h5'
muxH5FN = args.muxFile #'kspacemux.h5'#'ex4302_se11_01-03-2016_104856Mux.h5'
reduceValid = args.reduceValid

#Specific to the unaliasing
grappaTargetPacket = args.packet
grappaTargetSlice = args.slice
grappaTargetCoil = args.coil
smsR = args.smsR
fovShift = args.caipiShift

#Specific to the training
grappaBatch = args.batch
grappaEpochs = args.epoch
learningRate = args.learningRate
durationLimitFlag = args.epochsDefByDuration

#Specific to the network setup
networkLayers = args.networkLayers
kernelSize = args.kernelSize
numFilters = args.numFilters
middleChannels = args.middleChannels
dropOut = args.dropOut
batchNorm = args.batchNorm
groups = args.groups
biasFlag = args.bias

ssFlag = args.splitSlice
stdFlag = args.standardTrain

# # Define the dataset and the network

class ssGrappaDataset(Dataset):
    """Split slice GRAPPA dataset.
    TRAINING DATA: Aliased multi-channel complex in -> single slice, single coil
    The SS GRAPPA method essentially augments the training data with single band input->0 for subsets of slices in packet"""

    def __init__(self, muxK, calK, targetPacket, targetSlice, targetCoil):
        self.aliasedCplx = muxK.copy()
        self.calibrationCplx = calK.copy()
        
        self.makeRealData()
        
        self.cols = muxK.shape[3]
        self.rows = muxK.shape[2]
        self.coils = muxK.shape[1]
        self.packets = muxK.shape[0]
        self.smsR = calK.shape[0]
        
        self.targetPacket = targetPacket
        self.targetSlice = targetSlice
        self.targetCoil = targetCoil

        self.makeIteratorList()
        self.setupXY()
        
    def makeRealData(self):
        self.realAliased = np.concatenate( [np.real(self.aliasedCplx),\
                                            np.imag(self.aliasedCplx)], axis=1).astype(np.float32)
        self.realCal = np.concatenate( [np.real(self.calibrationCplx),\
                                        np.imag(self.calibrationCplx)], axis=2).astype(np.float32)
       
        #Below for normalization--disabled (mean=0, variance=1) for now
        self.aliasedMean = np.mean(self.realAliased) #enabled ASN 8.27
        self.aliasedVar = np.var(self.realAliased) #enabled ASN 8.27
        
        self.realAliased = ((self.realAliased-self.aliasedMean)/self.aliasedVar).astype(np.float32)
        
        self.calMean = 0#np.mean(self.realCal)
        self.calVar = 1#np.var(self.realCal)
        
        self.realCal = ((self.realCal-self.calMean)/self.calVar).astype(np.float32)
        
    def makeIteratorList(self):
        self.listOfIterators = []
        self.combinations=0
        self.listOfSliceIdcs = []
        for idx in range(1,self.smsR):
            self.listOfIterators.append(itertools.combinations(np.arange(self.smsR),idx))
        for iteratorInst in self.listOfIterators:
            for element in iteratorInst:
                tmpElement = list(element)
                self.listOfSliceIdcs.append(tuple(tmpElement))
                self.combinations+=1
    
    def __len__(self):
        return self.combinations
    
    def setupXY(self):
        #X is the aliased data--summed across list of slice idcs: 2*coils x rows x cols x calSets
        self.calX = np.zeros((self.realCal.shape[2],self.realCal.shape[3],self.realCal.shape[4],len(self)),\
                             dtype=np.float32)
        #Y is the ideal, single-slice, single-coil data: (2 (R/I) x rows x cols x calSets
        self.calY = np.zeros((2,self.realCal.shape[3], \
                              self.realCal.shape[4],len(self)),dtype=np.float32)
        #Loop and make the calibration/unaliased sets
        for idx in range(len(self)):
            #Sum across the aliased slices
            self.calX[::,::,::,idx] = np.sum(self.realCal[self.listOfSliceIdcs[idx],self.targetPacket,::,::,::],\
                                             axis=0, dtype=np.float32)
            self.calX[::,::,::,idx] /= float(len(self.listOfSliceIdcs[idx])) #ASN 8.27 norm
            #Set the ideal, single-slice, single-coil data to be non-zero if that slice/coil is in the aliased set
            if self.targetSlice in self.listOfSliceIdcs[idx]:
                self.calY[0,::,::,idx] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil,::,::]
                self.calY[1,::,::,idx] = self.realCal[self.targetSlice,self.targetPacket,\
                                                      self.targetCoil+self.coils,::,::]
                self.calY = self.calY.astype(np.float32)
    
    def __getitem__(self, idx):            
        return(self.calX[::,::,::,idx].astype(np.float32), self.calY[::,::,::,idx].astype(np.float32))


class grappaDataset(Dataset):
    """Standard GRAPPA dataset.
    TRAINING DATA: Aliased multi-channel complex in -> single slice, single coil
    The GRAPPA method does not include any augmentation"""

    def __init__(self, muxK, calK, targetPacket, targetSlice, targetCoil):
        self.aliasedCplx = muxK.copy()
        self.calibrationCplx = calK.copy()
        
        self.makeRealData()
        
        self.cols = muxK.shape[3]
        self.rows = muxK.shape[2]
        self.coils = muxK.shape[1]
        self.packets = muxK.shape[0]
        self.smsR = calK.shape[0]
        
        self.targetPacket = targetPacket
        self.targetSlice = targetSlice
        self.targetCoil = targetCoil
        
        self.setupXY()
        
    def makeRealData(self):
        self.realAliased = np.concatenate( [np.real(self.aliasedCplx), \
                                            np.imag(self.aliasedCplx)], axis=1).astype(np.float32)
        self.realCal = np.concatenate( [np.real(self.calibrationCplx), \
                                        np.imag(self.calibrationCplx)], axis=2).astype(np.float32)
        
        #Below for normalization--disabled (mean=0, variance=1)
        self.aliasedMean = 0#np.mean(self.realAliased)
        self.aliasedVar = 1#np.var(self.realAliased)
        
        self.realAliased = (self.realAliased-self.aliasedMean)/self.aliasedVar
        
        self.calMean = 0#np.mean(self.realCal)
        self.calVar = 1#np.var(self.realCal)
        
        self.realCal = (self.realCal-self.calMean)/self.calVar
        
        
    def __len__(self):
        return 1
    
    def setupXY(self):
        #X is the aliased data--summed across list of slice idcs: 2*coils x rows x cols
        self.calX = np.zeros((self.realCal.shape[2],self.realCal.shape[3],self.realCal.shape[4]), \
                             dtype=np.float32)
        #Y is the ideal, single-slice, single-coil data: (2 (R/I) x rows x cols
        self.calY = np.zeros((2,self.realCal.shape[3],\
                              self.realCal.shape[4]),dtype=np.float32)

        #Fill X by summing across all excited slices for this packet
        self.calX[::,::,::] = np.sum(self.realCal[::,self.targetPacket,::,::,::], \
                                     axis=0, dtype=np.float32)
        #Grab the ideal slice, packet, coil real and imaginary data
        self.calY[0,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil,::,::]
        self.calY[1,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil+self.coils,::,::]
        
    def __getitem__(self, idx):            
        return(self.calX[::,::,::].astype(np.float32), self.calY[::,::,::].astype(np.float32))


class inferenceDataset(Dataset):
    """Dataset for inference with GRAPPA, SS GRAPPA, RAKI, and SS RAKI."""

    def __init__(self, muxK, calK, targetPacket, targetSlice, targetCoil):
        self.aliasedCplx = muxK.copy()
        self.calibrationCplx = calK.copy()
        
        self.smsR = calK.shape[0] #moved here ASN 8.27
        self.aliasedCplx = self.aliasedCplx / float(self.smsR) #ASN 8.27

        self.makeRealData()
        
        self.phases = muxK.shape[0]
        self.packets = muxK.shape[1]
        self.coils = muxK.shape[2]
        self.rows = muxK.shape[3]
        self.cols = muxK.shape[4]
        
        self.targetPacket = targetPacket
        self.targetSlice = targetSlice
        self.targetCoil = targetCoil
        
        #Set up Y as the ideal real and imaginary data for this slice, packet, coil
        self.calY = np.zeros((2,self.realCal.shape[3], \
                             self.realCal.shape[4]),dtype=np.float32)
        self.calY[0,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil,::,::]
        self.calY[1,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil+self.coils,::,::]
        
    def makeRealData(self):
        self.realAliased = np.concatenate( [np.real(self.aliasedCplx), \
                                            np.imag(self.aliasedCplx)], axis=2).astype(np.float32)
        self.realCal = np.concatenate( [np.real(self.calibrationCplx), \
                                                np.imag(self.calibrationCplx)], axis=2).astype(np.float32)
        
        #Below for normalization--disabled (mean=0, variance=1) for now
        self.aliasedMean = np.mean(self.realAliased) #enabled ASN 8.27
        self.aliasedVar = np.var(self.realAliased) #enabled ASN 8.27
        
        self.realAliased = (self.realAliased-self.aliasedMean)/self.aliasedVar
        
        self.calMean = 0#np.mean(self.realCal)
        self.calVar = 1#np.var(self.realCal)
        
        self.realCal = (self.realCal-self.calMean)/self.calVar
    
    def __len__(self):
        return self.phases
    
    def __getitem__(self, idx):            
        #The ideal is known from the provided calibraiton array, and the aliased is input
        return(self.realAliased[idx,self.targetPacket,::,::,::].astype(np.float32), self.calY[::,::,::].astype(np.float32))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def makeRakiNetwork(inChannels,kernSize,midChannels,networkLayers,dropOut,batchNorm,groups,biasFL,numFilters):
    #Works well with 
    # L1: inChannels->inChannels; kernel 7; no bias; BatchNorm; ReLU; Dropout (.5)
    # L2: inChannels->512; kernel 1; no bias; BatchNorm; ReLU; Dropout (.5)
    # L3: 512->inChannels*numSlices; kernel 7; no bias
    #kernSize = 7
    #midChannels = 1024

    netList = []
    outputChan=inChannels
    for idx in range(networkLayers):
        #Set the input channel count equal to the output channel count of last layer
        inChan = outputChan
        #Final output has 2 channels for real/imag
        if idx+1 == networkLayers:
            outputChan = 2
            thisKern = kernSize
        #Penultimate output has midChannels
        elif idx+2 == networkLayers:
            outputChan = midChannels
            thisKern = 1
        #Input the number of in channels, output number of filters if first layer
        elif idx == 0:
            outputChan = numFilters
            thisKern = kernSize
        #All other layers keep number of channels constant
        else:
            outputChan = inChannels
            thisKern = kernSize

        #Make the 2D convolution layer
        netList.append(nn.Conv2d(in_channels = inChan, \
                                 out_channels = outputChan, \
                                 kernel_size = thisKern, \
                                 stride = 1, \
                                 padding = int(np.floor(thisKern/2.)), \
                                 dilation = 1, \
                                 groups = groups, \
                                 bias = biasFL, \
                                 padding_mode='zeros'))
        #Make the batch normalizaiton layer (if desired)
        if ((batchNorm) and (idx<networkLayers-1)):
            netList.append(nn.BatchNorm2d(outputChan))
        #Make the activation layer (only if not last layer of network)
        if idx+1 < networkLayers:
            netList.append(nn.ReLU())
            #Add the dropout layer
            netList.append(nn.Dropout2d(dropOut))

    rakiCNN = nn.Sequential(*netList)
    rakiCNN.apply(weights_init)

    return rakiCNN

def FTReco(inArray,dim0=0,dim1=1):
    outArray = np.fft.fftshift(np.fft.fft(np.fft.fftshift(inArray,axes=dim0),axis=dim0),axes=dim0)
    outArray = np.fft.fftshift(np.fft.fft(np.fft.fftshift(outArray,axes=dim1),axis=dim1),axes=dim1)
    return outArray

def IFTReco(inArray,dim0=0,dim1=1):
    outArray = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(inArray,axes=dim0),axis=dim0),axes=dim0)
    outArray = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(outArray,axes=dim1),axis=dim1),axes=dim1)
    return outArray

def makeSOS(inputArr,chanDim=2):
    calSOS = (inputArr * np.conjugate(inputArr))
    calSOS = np.sum(calSOS,axis=chanDim)
    calSOS = np.sqrt(calSOS)
    return(calSOS)

# #####################
# Get down to BUSINESS#
# #####################

# Read the data

calH5 = h5.File(os.path.join(dataDir,calH5FN), 'r')
muxH5 = h5.File(os.path.join(dataDir,muxH5FN), 'r')

calKRaw = calH5['kspaceCalUnalias']
muxKRaw = muxH5['kspacemux']

calK = calKRaw['real'] + 1j*calKRaw['imag'] #Dimensions--SMS slices, shots, coils, rows, cols
muxK = muxKRaw['real'] + 1j*muxKRaw['imag'] #Dimensions--phases, shots, coils, rows, cols

calKRaw = []
muxKRaw = []

if reduceValid:
    muxK = muxK[0::20,::,::,::,::]

calH5.close()
muxH5.close()


# Make the CAIPI shifted training data

calImgShift = FTReco(calK,dim0=-2,dim1=-1).astype(np.complex64)

calImgShift = np.roll(calImgShift,4,axis=0)
calImgShift = np.flip(calImgShift,axis=0)

calRows = calImgShift.shape[-2]
rowStep = int(np.floor(calRows/fovShift))*(1) #ASN--FOV shifts seem to switch +/- :-(
for slc in range(calImgShift.shape[0]):
    slcFac = slc
    while slcFac >= fovShift:
        slcFac -= fovShift
    startRow = -1*slcFac*rowStep
    
    if startRow != 0:
        arr1 = calImgShift[slc,::,::,startRow::,::].copy()
        arr2 = calImgShift[slc,::,::,0:startRow:,::].copy()
        calImgShift[slc,::,::,::,::] = np.concatenate([arr1,arr2],axis=-2)

calKShift = IFTReco(calImgShift,dim0=-2,dim1=-1).astype(np.complex64)


# Instantiate the training datasets

#Split slice set
if ssFlag:
    thisSSGrappaDataset = ssGrappaDataset(calK=calKShift, muxK=muxK[0,::,::,::,::], \
                                          targetPacket=grappaTargetPacket, \
                                          targetSlice=grappaTargetSlice, \
                                          targetCoil=grappaTargetCoil)
#Standard set
if stdFlag:
    thisGrappaDataset = grappaDataset(calK=calKShift, muxK=muxK[0,::,::,::,::], \
                                      targetPacket=grappaTargetPacket, \
                                      targetSlice=grappaTargetSlice, \
                                      targetCoil=grappaTargetCoil)


# Make the network, optimizer, and cost function

inChannels = 2*calImgShift.shape[2]

if ssFlag:
    thisSSCNN = makeRakiNetwork(inChannels, \
                                kernelSize, \
                                middleChannels, \
                                networkLayers, \
                                dropOut, \
                                batchNorm, \
                                groups, \
                                biasFlag, \
                                numFilters ).cuda()
    ssCostFN = nn.L1Loss().cuda()
    ssOptimizer = torch.optim.Adam(thisSSCNN.parameters(),lr=learningRate)

if stdFlag:
    thisCNN = makeRakiNetwork(inChannels, \
                              kernelSize, \
                              middleChannels, \
                              networkLayers, \
                              dropOut, \
                              batchNorm, \
                              groups, \
                              biasFlag, \
                              numFilters ).cuda()
    costFN = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(thisCNN.parameters(),lr=learningRate)

# # TRAIN!

nvidiaSmiCmd = 'nvidia-smi -q -d MEMORY -d UTILIZATION -i 0'

if ssFlag:
    print('Split-Slice Network Training')
    print(thisSSCNN)
    ssErrors = []
    startTime=time.time()
    ep = 0
    myContinue = True
    while myContinue:
    #for ep in range(grappaEpochs):
        ep+=1
        thisSSCNN.train(True)
    
        dataLoader = DataLoader(thisSSGrappaDataset, 
                                batch_size=int(grappaBatch),
                                shuffle=True, 
                                num_workers=0)
        for i_batch, sample_batched in enumerate(dataLoader):
            #Run the prediction
            #thisSSCNN.half()
            sampleX = sample_batched[0].cuda()
            yPred = thisSSCNN(sampleX)
            #yPred = yPred.float()
    
            #Compute the loss
            sampleY = sample_batched[1].cuda()
            #sampleY = sampleY.float()
            loss = ssCostFN(yPred, sampleY)
    
            #Zero the gradients
            ssOptimizer.zero_grad()
    
            #Do the back propogation
            loss.backward()
    
            #Take the next step
            #thisSSCNN.float()
            ssOptimizer.step()
        ssErrors.append(loss.item())
        duration = time.time()-startTime
        if ep%10 == 0:
            print('Duration %f s'%(duration))
            print('    %i %f'%(ep, loss.item(),))
        if durationLimitFlag:
            if duration > grappaEpochs:
                myContinue=False
        else:
            if ep >= grappaEpochs:
                myContinue=False
        ssEpRun = ep
    ssDur = time.time()-startTime
    
    #Get NVIDIA-SMI stats
    ssNVSmiOutput = subprocess.getoutput(nvidiaSmiCmd)
    
if stdFlag:
    print('Non-Augmented Network Training')
    print(thisCNN)
    errors = []
    startTime=time.time()
    ep = 0
    myContinue=True
    while myContinue:
    #for ep in range(grappaEpochs):
        ep+=1
        thisCNN.train(True)
    
        dataLoader = DataLoader(thisGrappaDataset, 
                                batch_size=1,
                                shuffle=True, 
                                num_workers=0)
        for i_batch, sample_batched in enumerate(dataLoader):
            #Run the prediction
            #thisCNN.half()
            sampleX = sample_batched[0].cuda()
            yPred = thisCNN(sampleX) 
            #yPred = yPred.float()
    
            #Compute the loss
            sampleY = sample_batched[1].cuda()
            #sampleY = sampleY.float()
            loss = costFN(yPred, sampleY)
    
            #Zero the gradients
            optimizer.zero_grad()
    
            #Do the back propogation
            loss.backward()
    
            #Take the next step
            #thisCNN.float()
            optimizer.step()
        errors.append(loss.item())
        duration = time.time()-startTime
        if ep%1000 == 0:
            print('Duration %f s'%(duration))
            print('    %i %f'%(ep, loss.item(),))
        if durationLimitFlag:
            if duration > grappaEpochs:
                myContinue=False
        else:
            if ep >= grappaEpochs:
                myContinue=False
        stdEpRun = ep
    stdDur = time.time()-startTime
    
    stdNVSmiOutput = subprocess.getoutput(nvidiaSmiCmd)

#Save the results of training
dirName='sd_%i_cf_%s_mf_%s_p_%i_s_%i_c_%i_r_%i_ba_%i_e_%i_lr_%f_nl_%i_ks_%i_mc_%i_do_%f_bn_%i_g_%i_bi_%i_rv_%i_cs_%i_nf_%i'%\
         (seed, \
          calH5FN, \
          muxH5FN, \
          grappaTargetPacket, \
          grappaTargetSlice, \
          grappaTargetCoil, \
          smsR, \
          grappaBatch, \
          grappaEpochs, \
          learningRate, \
          networkLayers, \
          kernelSize, \
          middleChannels, \
          dropOut, \
          batchNorm, \
          groups, \
          biasFlag, \
          reduceValid, \
          fovShift, \
          numFilters)
dirName=os.path.join('NenckaCluster',dirName)
try:
    os.makedirs(dirName)
except:
    print('Directory %s exists or cannot be made'%(dirName))

if ssFlag:
    #Save SS results
    torch.save(thisSSCNN.state_dict(),os.path.join(dirName,'thisSSCNN.state_dict'))
    with open(os.path.join(dirName,'ssError.pkl'),'wb') as f:
        pickle.dump(ssErrors,f)
    plt.plot(np.arange(0,0.999999999999,1./len(ssErrors)),ssErrors)
    plt.savefig(os.path.join(dirName,'ssError.png'))
    open(os.path.join(dirName,'%s.ssDur'%(ssDur)),'a').close()
    open(os.path.join(dirName,'%i.ssEpRun'%(ssEpRun)),'a').close()
    with open(os.path.join(dirName,'ssNVSMI.out'),'w') as f:
        f.write(ssNVSmiOutput)

if stdFlag:
    #Save non-augmented results
    torch.save(thisCNN.state_dict(),os.path.join(dirName,'thisCNN.state_dict'))
    with open(os.path.join(dirName,'error.pkl'),'wb') as f:
        pickle.dump(errors,f)
    plt.plot(np.arange(0,0.999999999999,1./len(errors)),errors)
    plt.savefig(os.path.join(dirName,'stdError.png'))
    open(os.path.join(dirName,'%s.stdDur'%(stdDur)),'a').close()
    open(os.path.join(dirName,'%i.stdEpRun'%(stdEpRun)),'a').close()
    with open(os.path.join(dirName,'stdNVSMI.out'),'w') as f:
        f.write(stdNVSmiOutput)

# Do validation runs!
thisValidDataset = inferenceDataset(calK=calKShift, muxK=muxK[::,::,::,::,::], \
                                    targetPacket=grappaTargetPacket, \
                                    targetSlice = grappaTargetSlice, \
                                    targetCoil = grappaTargetCoil)

if ssFlag:
    # First validate with split-slice
    ssInferred = []
    thisSSCNN.train(False) 
    with torch.no_grad(): #Needed to avoid memory errors!!!
        dataLoader = DataLoader(thisValidDataset, 
                                batch_size=muxK.shape[0],
                                shuffle=False, 
                                num_workers=0)
        for i_batch, sample_batched in enumerate(dataLoader):
            #Run the prediction
            sampleX = sample_batched[0].cuda()
            yPred = thisSSCNN(sampleX)
    
            ssInferred.append(yPred.cpu().detach().numpy())
            
            loss = ssCostFN(yPred, sample_batched[1].cuda())
    ssValidLoss = loss.item()
    
    #Make the actual images
    ssYPredOut = np.zeros((len(thisValidDataset),\
                          yPred.shape[2],yPred.shape[3]),dtype=np.complex64)
    ssYPredOut *= thisValidDataset.calVar
    ssYPredOut += thisValidDataset.calMean
    
    for idx0 in range(len(ssInferred)):
        ssYPredOut[idx0*ssInferred[0].shape[0]:(idx0+1)*ssInferred[0].shape[0],::,::] =\
                  ssInferred[idx0][::,0,::,::] + \
                  1j*ssInferred[idx0][::,1,::,::]
    
    ssYPredOut = FTReco(ssYPredOut,dim0=-2,dim1=-1)

if stdFlag:
    # Then validate with non-augmented
    inferred = []
    thisCNN.train(False) 
    with torch.no_grad(): #Needed to avoid memory errors!!!
        dataLoader = DataLoader(thisValidDataset, 
                                batch_size=muxK.shape[0],
                                shuffle=False, 
                                num_workers=0)
        for i_batch, sample_batched in enumerate(dataLoader):
            #Run the prediction
            sampleX = sample_batched[0].cuda()
            yPred = thisCNN(sampleX)
    
            inferred.append(yPred.cpu().detach().numpy())
            
            loss = costFN(yPred, sample_batched[1].cuda())
    validLoss = loss.item()
    
    #Make the actual images
    yPredOut = np.zeros((len(thisValidDataset),\
                        yPred.shape[2],yPred.shape[3]),dtype=np.complex64)
    yPredOut *= thisValidDataset.calVar
    yPredOut += thisValidDataset.calMean
    
    for idx0 in range(len(inferred)):
        yPredOut[idx0*inferred[0].shape[0]:(idx0+1)*inferred[0].shape[0],::,::] =\
                 inferred[idx0][::,0,::,::] + \
                 1j*inferred[idx0][::,1,::,::]
    
    yPredOut = FTReco(yPredOut,dim0=-2,dim1=-1)
    

# Write the validation results

if ssFlag:
    #Save SS results
    plt.figure(figsize=(20,20))
    plt.title('SS')
    for rep in range(min(ssYPredOut.shape[0],20)):
        plt.subplot(5,4,rep+1)
        plt.imshow(np.abs(ssYPredOut[rep,::,::]))
        plt.title("%i"%(rep))
    plt.savefig(os.path.join(dirName,'ssValidImg.png'))
    open(os.path.join(dirName,'%s.ssVal'%(ssValidLoss)),'a').close()
    np.save(os.path.join(dirName,'ssYPredOut.npy'),ssYPredOut)

if stdFlag:
    #Save non-augmented results
    plt.figure(figsize=(20,20))
    plt.title('StD')
    for rep in range(min(yPredOut.shape[0],20)):
        plt.subplot(5,4,rep+1)
        plt.imshow(np.abs(yPredOut[rep,::,::]))
        plt.title("%i"%(rep))
    plt.savefig(os.path.join(dirName,'stdValidImg.png'))
    open(os.path.join(dirName,'%s.stdVal'%(validLoss)),'a').close()
    np.save(os.path.join(dirName,'stdYPredOut.npy'),yPredOut)

