import numpy as np

def getKernFromPoint(inArray, pointTupple, kernel):
    kernRows,kernCols,kernCoils = np.shape(kernel)
    tmpKernelVals = inArray[int(pointTupple[0]-np.floor(kernRows/2)):\
                            int(pointTupple[0]+np.floor(kernRows/2)+1), \
                            int(pointTupple[1]-np.floor(kernCols/2)):\
                            int(pointTupple[1]+np.floor(kernCols/2)+1), \
                            ::] \
                    * kernel
    tmpKernelVec = np.zeros((kernRows*kernCols*kernCoils),dtype=np.complex64)
    idx=0
    for coilNo in range(kernCoils):
        for colNo in range(kernCols):
            for rowNo  in range(kernRows):
                tmpKernelVec[idx] = tmpKernelVals[rowNo,colNo,coilNo]
                idx+=1
    return tmpKernelVec

def makeKernSolvingArrays(idealArray, aliasedArray, kernel):
    idealRows,idealCols = np.shape(idealArray)
    aliasedRows,aliasedCols,aliasedCoils = np.shape(aliasedArray)
    kernRows,kernCols,kernCoils = np.shape(kernel)
    
    fitPoints = (idealRows-kernRows+1)*(idealCols-kernCols+1)
    
    kernDM = np.zeros((fitPoints,kernRows*kernCols*kernCoils),dtype=np.complex64)
    kernY = np.zeros((fitPoints,1),dtype=np.complex64)
    
    idx=0
    for rowNo in range(aliasedRows-kernRows+1):
        rowVal = rowNo+int(kernRows/2)
        
        for colNo in range(aliasedCols-kernRows+1):
            colVal = colNo+int(kernCols/2)
            
            kernY[idx,0] = idealArray[rowVal,colVal]
            kernDM[idx,::] = getKernFromPoint(aliasedArray,(rowVal,colVal),kernel)
            idx+=1
            
    return kernY,kernDM

def applyGrappaKernel(aliasedArray,kernelVec,kernelPoints):
    aliasedRows,aliasedCols,aliasedCoils = np.shape(aliasedArray)
    kernRows,kernCols,kernCoils = np.shape(kernelPoints)
    unaliasedK = np.zeros((aliasedRows,aliasedCols),dtype=np.complex64)
    
    for rowNo in range(aliasedRows-kernRows+1):
        rowVal = rowNo+int(kernRows/2)
        
        for colNo in range(aliasedCols-kernRows+1):
            colVal = colNo+int(kernCols/2)
            
            tmpKernWts = getKernFromPoint(aliasedArray,(rowVal,colVal),kernelPoints)
            tmpVal = np.dot(kernelVec[::,0], tmpKernWts)
            unaliasedK[rowVal,colVal] = tmpVal
            
    return unaliasedK

def grappaUnaliasSeries(idealArray,aliasedTrainArray,aliasedTimeSeries,\
                        kernel,regularize=False,regVal=0.02, unalCoil=None):
    idealRows,idealCols,idealSlcs,idealCoils,idealReps = np.shape(idealArray)
    aliasedRows,aliasedCols,aliasedCoils,aliasedReps = np.shape(aliasedTimeSeries)
    kernRows,kernCols,kernCoils = np.shape(kernel)
    
    fitPointsPerRep = (idealRows-kernRows+1)*(idealCols-kernCols+1)
    fitPoints = fitPointsPerRep*idealReps
    
    unaliasedK = np.zeros((aliasedRows,aliasedCols,idealSlcs,aliasedCoils,aliasedReps),\
                          dtype=np.complex64)
    
    if unalCoil!=None:
        coilRange = [unalCoil]
    else:
        coilRange = np.arrange(0,idealCoils,1)
    
    for coilNo in coilRange:
        for sliceNo in range(idealSlcs):
            print('GRAPPA Solving coil %i and slice %i'%(coilNo,sliceNo))
            #Start by solving the kernel for this coil/slice combination
            kernY = np.zeros((fitPoints,1),dtype=np.complex64)
            kernDM = np.zeros((fitPoints,kernRows*kernCols*kernCoils),dtype=np.complex64)
            for idealRep in range(idealReps):
                tmpKernY,tmpKernDM = makeKernSolvingArrays( \
                                       idealArray[::,::,sliceNo,coilNo,idealRep], \
                                       aliasedTrainArray[::,::,::,idealRep], \
                                       kernel) #ASN test train with 1 back phantom
                kernY[idealRep*fitPointsPerRep:(idealRep+1)*fitPointsPerRep] = \
                    tmpKernY.copy()
                kernDM[idealRep*fitPointsPerRep:(idealRep+1)*fitPointsPerRep, ::] = \
                    tmpKernDM.copy()
            if not regularize:
                kernSoln, _, _, _ = np.linalg.lstsq(kernDM, kernY,rcond=None)
            else:
                AtA = kernDM.conj().T @ kernDM #Note @ denotes matrix multiplication
                #Use TK regularization per Stanford who quotes Lustig
                lambdaVal = np.linalg.norm(AtA, ord='fro')/np.shape(AtA)[1] * regVal
                if False:
                    print(kernDM.shape)
                    print(AtA.shape)
                    print(np.eye(np.shape(AtA)[0]).shape)
                    print(lambdaVal.shape)
                    print(kernY.shape)
                kernSoln = np.linalg.inv(AtA + np.eye(np.shape(AtA)[0])*lambdaVal) @ \
                    kernDM.conj().T @ kernY
            #Then apply the kernel to the acquired data
            for aliasedRep in range(aliasedReps):
                unaliasedK[::,::,sliceNo,coilNo,aliasedRep] = \
                    applyGrappaKernel(aliasedTimeSeries[::,::,::,aliasedRep], \
                                      kernSoln,kernel)
    return unaliasedK

#### Split-slice grappa
def getSSKernFromPoint(idealSmsArray, pointTupple, kernel):
    kernRows,kernCols,kernCoils = np.shape(kernel)
    idealRows,idealCols,idealSlcs,idealCoils = np.shape(idealSmsArray)
    
    tmpKernelVals = np.zeros((kernRows,kernCols,kernCoils,idealSlcs),dtype=np.complex64)
    for slcNo in range(idealSlcs):
        tmpKernelVals[::,::,::,slcNo] = \
            idealSmsArray[int(pointTupple[0]-np.floor(kernRows/2)): \
                          int(pointTupple[0]+np.floor(kernRows/2)+1), \
                          int(pointTupple[1]-np.floor(kernCols/2)): \
                          int(pointTupple[1]+np.floor(kernCols/2)+1), \
                          slcNo, ::] \
            * kernel
    
    if False:
        print("Shape of tmpKernelVals")
        print(np.shape(tmpKernelVals))
    
    tmpKernelArray = np.zeros((idealSlcs,kernRows*kernCols*kernCoils),dtype=np.complex64)
    for slcNo in range(idealSlcs):
        idx = 0
        for coilNo in range(kernCoils):
            for colNo in range(kernCols):
                for rowNo in range(kernRows):
                    tmpKernelArray[slcNo,idx] = tmpKernelVals[rowNo,colNo,coilNo,slcNo]
                    idx += 1
    return tmpKernelArray

def makeSSKernSolvingArrays(idealSmsArray, kernel, thisSliceNo, thisCoilNo):
    kernRows,kernCols,kernCoils = np.shape(kernel)
    idealRows,idealCols,idealSlcs,idealCoils = np.shape(idealSmsArray)
    
    fitPoints = (idealRows-kernRows+1)*(idealCols-kernCols+1)
    
    kernDM = np.zeros((fitPoints*idealSlcs,kernRows*kernCols*kernCoils),dtype=np.complex64)
    kernY = np.zeros((fitPoints*idealSlcs,1),dtype=np.complex64)
    
    idx=0
    for rowNo in range(idealRows-kernRows+1):
        rowVal = rowNo+int(kernRows/2)
        
        for colNo in range(idealCols-kernRows+1):
            colVal = colNo+int(kernCols/2)
            
            kernY[idx+thisSliceNo,0] = idealSmsArray[rowVal,colVal,thisSliceNo,thisCoilNo]
            kernDM[idx:idx+idealSlcs,::] = getSSKernFromPoint(idealSmsArray,(rowVal,colVal),kernel)
            idx+=idealSlcs
            
    return kernY,kernDM

def applySSGrappaKernel(origAliasedArray,kernelArray,kernelPoints):
    #aliasedArray = np.transpose(origAliasedArray,(0,1,2,4,3)) #[rows,cols,coils]->[rows,cols,slcs,coils]
    aliasedArray = np.expand_dims(origAliasedArray,axis=2)
    if False:
        print('applySSGrappaKernel: aliasedArray shape:')
        print(np.shape(aliasedArray))
        print('kernelArray shape:')
        print(np.shape(kernelArray))
    
    aliasedRows,aliasedCols,aliasedSlcs,aliasedCoils = np.shape(aliasedArray)
    kernRows,kernCols,kernCoils = np.shape(kernelPoints)
    
    unaliasedK = np.zeros((aliasedRows,aliasedCols),dtype=np.complex64)
    
    for rowNo in range(aliasedRows-kernRows+1):
        rowVal = rowNo+int(kernRows/2)
        
        for colNo in range(aliasedCols-kernRows+1):
            colVal = colNo+int(kernCols/2)
            
            tmpKernWts = getSSKernFromPoint(aliasedArray,(rowVal,colVal),kernelPoints)
            tmpVal = np.dot(kernelArray[::,0], tmpKernWts[0,::])
            unaliasedK[rowVal,colVal] = tmpVal
            
    return unaliasedK

def ssGrappaUnaliasSeries(idealArray,aliasedTimeSeries,kernelSS,regularize=False,regVal=0.02):
    idealRows,idealCols,idealSlcs,idealCoils,idealReps = np.shape(idealArray)
    aliasedRows,aliasedCols,aliasedCoils,aliasedReps = np.shape(aliasedTimeSeries)
    kernRows,kernCols,kernCoils = np.shape(kernelSS)
    
    fitPointsPerRep = (idealRows-kernRows+1)*(idealCols-kernCols+1)*idealSlcs
    fitPoints = fitPointsPerRep*idealReps
    
    ssUnaliasedK = np.zeros((aliasedRows,aliasedCols,idealSlcs,aliasedCoils,aliasedReps),\
                          dtype=np.complex64)
    
    for coilNo in range(idealCoils):
        for sliceNo in range(idealSlcs):
            print('SS-GRAPPA Solving coil %i and slice %i'%(coilNo,sliceNo))
            
            kernY = np.zeros((fitPoints,1),dtype=np.complex64)
            kernDM = np.zeros((fitPoints,kernRows*kernCols*kernCoils),dtype=np.complex64)
            for idealRep in range(idealReps):
                tmpKernY,tmpKernDM = makeSSKernSolvingArrays( \
                               idealArray[::,::,::,::,idealRep], kernelSS, sliceNo, coilNo)
                kernY[idealRep*fitPointsPerRep:(idealRep+1)*fitPointsPerRep] = \
                    tmpKernY.copy()
                kernDM[idealRep*fitPointsPerRep:(idealRep+1)*fitPointsPerRep, ::] = \
                    tmpKernDM.copy()
            if not regularize:
                kernSoln, _, _, _ = np.linalg.lstsq(kernDM, kernY,rcond=None)
            else:
                AtA = kernDM.conj().T @ kernDM #Note @ denotes matrix multiplication
                #Use TK regularization per Stanford who quotes Lustig
                lambdaVal = np.linalg.norm(AtA, ord='fro')/np.shape(AtA)[1] * regVal
                if False:
                    print(kernDM.shape)
                    print(AtA.shape)
                    print(np.eye(np.shape(AtA)[0]).shape)
                    print(lambdaVal.shape)
                    print(kernY.shape)
                kernSoln = np.linalg.inv(AtA + np.eye(np.shape(AtA)[0])*lambdaVal) @ \
                    kernDM.conj().T @ kernY
                
            for phantomNo in range(aliasedReps):
                ssUnaliasedK[::,::,sliceNo,coilNo,phantomNo] = \
                    applySSGrappaKernel(aliasedTimeSeries[::,::,::,phantomNo], \
                                        kernSoln,kernelSS)
    return ssUnaliasedK