from scipy.spatial.distance import pdist,squareform
import numpy as np
import time
import pandas as pd
import numpy.matlib
from math import exp,ceil
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
import scipy.io as sio
from fisher_score import fisher_score
from itertools import cycle

global label_ratio 
global labeling_budget
label_ratio = 10
labeling_budget = 0
# Read Data from the csv. files
def Input():
    data_dir = pjoin(dirname(sio.__file__),
                     'C:/Users/Xuyang/Desktop/Spring-2021-PhD-Xuyang/Research/Fault-Detection-Self-Driving-Cars/Datasets/CARAL-Simulator/Town-03/SS/')
    csv_fname = pjoin(data_dir, 'CarlaTown03-30Vehicles-ML-fault-SS-2.csv')
    # sample = pd.read_csv(csv_fname,header=None)
    sample = pd.read_csv(csv_fname)
    attributes = sample.columns.values
    [N,L] = np.shape(sample)
    # Extract the label of the data from the data frame
    data, label = sample.iloc[:,0:L-4].values, sample.iloc[:,L-1].values
    return data, label, attributes
# Parameter Specification:
def ParamSpe(data):
    Buffersize = 1000 # set the size of the data chunk
    PreStd = [] # Initialize the summary vector of variance for the data stream
    normal_summary = [] # normal cluster summary
    fault_summary = [] # fault cluster summary
    PFS = [] # Initialize the summary vector to keep track of density values for cluster centers
    # Calculate the total number of chunks
    T = int(round(np.shape(data)[0] / Buffersize))
    return Buffersize, normal_summary, fault_summary, T, PFS, PreStd
# Perform the distance calculation
def Distance_Cal(data):
    D = pdist(data)
    Dist = squareform(D)
    return Dist
# Fitness Evaluation
def Fitness_Cal(sample,pop,stdData,gamma):
    Ns = np.shape(sample)[0]
    Np = np.shape(pop)[0]
    if Np == 0:
        return []
    Newsample = np.concatenate([sample,pop])
    Dist = Distance_Cal(Newsample)
    fitness = []
    for i in range(Np):
        distArray = np.power(Dist[i+Ns,0:Ns],2)
        temp = np.power(np.exp(-distArray/stdData),gamma)
        fitness.append(np.sum(temp))
    return np.asarray(fitness)
# Fitness Update from Historical Summary
def fitness_update(P_Summary, Current, fitness, PreStd, gamma, stdData):
    [N, dim] = np.shape(Current)
    t_I = len(PreStd)
    NewFit = fitness
    PreFit = P_Summary[:, dim]
    PreP = P_Summary[:, 0:dim]
    OldStd = PreStd[t_I - 1]
    dist_PreP = squareform(pdist(np.concatenate([Current, PreP])))
    if len(P_Summary) > 0:
        for i in range(N):
            fitin = 0
            tempdist_PreP = dist_PreP[i, N:]
            for j in range(np.shape(PreP)[0]):
                if tempdist_PreP[j] < 0.01:
                    fitin = PreFit[j]
                    break
                else:
                    d = tempdist_PreP[j]
                    fitin += (exp(-d ** 2 / stdData) ** gamma) * (PreFit[j] ** (OldStd / stdData))
            NewFit[i] = fitness[i] + fitin
    return NewFit
# Initialize the candidate set for searching temporary potential clusters (TPCs)
def PopInitial(sample, PreMu, PreStd, Buffersize):
    [n, l] = np.shape(sample)
    pop_Size = round(1 * n)
    # Compute the statistics of the current data chunk
    minLimit = np.min(sample, axis=0)
    meanData = np.mean(sample, axis=0)
    maxLimit = np.max(sample, axis=0)
    # Update the statistics of the data stream
    meanData = UpdateMean(PreMu, meanData)
    PreMu.append(meanData)
    # Compute the standard deviation of the current data chunk
    MD = np.matlib.repmat(meanData, n, 1)
    tempSum = np.sum(np.sum((MD - sample) ** 2, axis=1))
    stdData = tempSum / n
    # Update the standard deviation of the data stream
    stdData = StdUpdate(stdData, PreStd)
    # Randonmly Initialize the population indices from the data chunk
    pop_Index = np.arange(0, n)
    pop = sample[pop_Index, :]
    # Calculate the initial niche radius
    radius = numpy.linalg.norm((maxLimit - minLimit)) * 0.1
    if radius <= 0.01:
        radius = 1
    return [stdData, pop_Index, pop, radius, PreMu, PreStd]
# Update the mean of the data stream as new data chunk arrives
def UpdateMean(PreMu, meanData):
    # Num of the processed data chunk
    t_P = len(PreMu)
    # Update the mean of the data stream as new data chunk arrives
    if t_P == 0:
        newMu = meanData
    else:
        oldMu = PreMu[t_P - 1][:]
        newMu = (meanData + oldMu * t_P) / (t_P + 1)
    return newMu
# Update the variance of the data stream as new data chunk arrives
def StdUpdate(Std, PreStd):
    # Num of the processed data chunk
    t_P = len(PreStd)
    # Update the variance of the data stream as new data chunk arrives
    if t_P == 0:
        newStd = Std
    else:
        oldStd = PreStd[t_P - 1]
        newStd = (Std + oldStd * t_P) / (t_P + 1)
    return newStd
# ------------------------Parameter Estimation for gamma----------------------------#
def CCA(sample, stdData, Dist):
    m = 1
    gamma = 5
    ep = 0.985
    N = np.shape(sample)[0]
    while 1:
        den1 = []
        den2 = []
        for i in range(N):
            Diff = np.power(Dist[i, :], 2)
            temp1 = np.power(np.exp(-Diff / stdData), gamma * m)
            temp2 = np.power(np.exp(-Diff / stdData), gamma * (m + 1))
            den1.append(np.sum(temp1))
            den2.append(np.sum(temp2))
        y = np.corrcoef(den1, den2)[0, 1]
        if y > ep:
            break
        m = m + 1
    return m * gamma
def DCCA(sample,stdData,P_Summary,gamma,dim):
    P_Center = P_Summary[:,0:dim] # Historical cluster centers in the cluster summary
    P_F = P_Summary[:,dim] # Density values for historical clusters
    gam1 = gamma # Gamm value at t-1
    N1, N2 = np.shape(sample)[0], np.shape(P_Center)[0]
    ep = 0.985 # Threshold value for correlation comparison 0.985
    N = N1 + N2
    # Concatenate samples and historical cluster centers together
    temp = np.concatenate([sample,P_Center],axis=0)
    # Distance calculation for the concatenated set
    Dist = Distance_Cal(temp)
    while 1:
        gam2 = gam1 + 5
        den1, den2 = [], []
        for i in range(N):
            Diff = np.power(Dist[i,0:N1],2)
            temp1 = np.power(np.exp(-Diff/stdData),gam1)
            temp2 = np.power(np.exp(-Diff/stdData),gam2)
            sum1, sum2 = np.sum(temp1), np.sum(temp2)
            if i < N1:
                T1, T2 = 0, 0
                for j in range(N2):
                    T1 += P_F[j]**(gam1/gamma)
                    T2 += P_F[j]**(gam2/gamma)
                s1, s2 = sum1 + T1, sum2 + T2
            else:
                s1 = sum1 + P_F[i-N1]**(gam1/gamma)
                s2 = sum2 + P_F[i-N1]**(gam2/gamma)
            den1.append(s1)
            den2.append(s2)
        y = np.corrcoef(den1,den2)[0,1]
        if y > ep:
            break
        gam1 = gam2
    return gam1
# Perform the TPC search among the population
def TPC_Search(Dist, Pop_Index, Pop, radius, fitness):
    # Extract the size of the population
    [N, dim] = np.shape(Pop)
    P = []  # Initialize the TPC Vector
    P_fitness = [] # Initialize the fitness vector for TPCs
    marked = [] # A vector to accumulate the indices of samples have been assigned to TPCs
    co = [] # The
    OriginalIndice = Pop_Index
    OriginalFit = fitness
    OriginalPop = Pop
    PeakIndice = []
    TPC_Indice = OriginalIndice
    while 1:
        # -------------Search for the local maximum-----------------#
        SortIndice = np.argsort(fitness)
        NewIndice = SortIndice[::-1]
        Pop = Pop[NewIndice, :]
        fitness = fitness[NewIndice]
        OriginalIndice = OriginalIndice[NewIndice]
        P.append(Pop[0, :])
        P_fitness.append(fitness[0])
        P_Indice = OriginalIndice[0]
        PeakIndice.append(np.where(OriginalFit == fitness[0])[0][0])
        Ind = AssigntoPeaks(Pop, Pop_Index, P, P_Indice, marked, radius, Dist)
        marked.append(Ind)
        marked.append(NewIndice[0])
        if not Ind:
            Ind = [NewIndice[0]]
            co.append(1)
        else:
            co.append(len(Ind))
        TempFit = fitness
        sum1 = 0
        TPC_Indice[Ind] = np.where(OriginalFit == fitness[0])[0][0]
        for j in range(len(Ind)):
            sum1 += fitness[np.where(OriginalIndice == Ind[j])]
        for th in range(len(Ind)):
            TempFit[np.where(OriginalIndice == Ind[th])] = fitness[np.where(OriginalIndice == Ind[th])] / (1 + sum1)
        fitness = TempFit
        if np.sum(co) >= N:
            P2 = OriginalPop[PeakIndice][:]
            P = np.asarray(P2)
            P_fitness = np.asarray(P_fitness)
            TPC_Indice = Close_Clusters(Pop, PeakIndice, Dist)
            break
    return P, P_fitness, TPC_Indice, PeakIndice
# Find the cluster indices for samples
def Close_Clusters(pop, PeakIndices, Dist):
    P = pop[PeakIndices][:]
    C_Indices = np.arange(0, np.shape(pop)[0])
    for i in range(np.shape(pop)[0]):
        temp_dist = Dist[i][PeakIndices]
        C_Indices[i] = PeakIndices[np.argmin(temp_dist)]
    return C_Indices
# Compute the closest cluster indice for all samples and return the minimum distance
def Cluster_Assign(sample, P):
    # Number of samples
    N = np.shape(sample)[0]
    # Number of Clusters at t
    Np = np.shape(P)[0]
    MinDist = []
    MinIndice = []
    if Np == 0:
        return [], []
    dist_toP = squareform(pdist(np.concatenate([P, sample], axis=0)))
    for i in range(N):
        d = dist_toP[i+Np, :Np]
        if len(d) <= 1:
            tempD = d
            tempI = 0
        else:
            tempD = np.min(d)
            tempI = np.argmin(d)
        MinDist.append(tempD)
        MinIndice.append(tempI)
    MinDist = np.asarray(MinDist)
    MinIndice = np.asarray(MinIndice)
    return MinDist, MinIndice
# Perform the merge of TPCs in the current data chunk Ct
def MergeInChunk(P, P_fitness, sample, gamma, stdData, Dist, TPC_Indice, PeakIndices):
    #-----------------Perform the Merge of TPCs witnin each data chunk----------------#
    # Num of TPCs
    [Nc, dim] = np.shape(P)
    NewP = []
    NewP_fitness = []
    NewPeakIndices = []
    marked = []
    unmarked = []
    Com = []
    # Num of TPCs
    Nc = np.shape(P)[0]
    for i in range(Nc):
        MinDist = np.inf
        MinIndice = 100000
        if i not in marked:
            for j in range(Nc):
                if j != i and j not in marked:
                    d = np.linalg.norm(P[j, :] - P[i, :])
                    if d < MinDist:
                        MinDist = d
                        MinIndice = j
            if MinIndice <= Nc:
                MinIndice = int(MinIndice)
                Merge = True
                Neighbor = PeakIndices[MinIndice]
                Current = PeakIndices[i]
                X = Boundary_Instance(Current, Neighbor, Dist, TPC_Indice, sample)
                X = np.reshape(X, (1, np.shape(P)[1]))
                fitX = Fitness_Cal(sample, X, stdData, gamma)
                fitP = P_fitness[i]
                fitN = P_fitness[MinIndice]
                if fitX < 0.95* min(fitN, fitP):  # 0.95 Aggregation
                    Merge = False
                else:
                    Merge = True
                if Merge:
                    Com.append([i, MinIndice])
                    marked.append(MinIndice)
                    marked.append(i)
                else:
                    unmarked.append(i)
    Com = np.asarray(Com)
    # Number of Possible Merges:
    Nm = np.shape(Com)[0]
    for k in range(Nm):
        if P_fitness[Com[k, 0]] >= P_fitness[Com[k, 1]]:
            NewPeakIndices.append(PeakIndices[Com[k, 0]])
        else:
            NewPeakIndices.append(PeakIndices[Com[k, 1]])
    # Add Unmerged TPCs to the NewP
    for n in range(Nc):
        if n not in Com:
            NewPeakIndices.append(PeakIndices[n])
    NewPeakIndices = np.unique(NewPeakIndices)
    NewP = sample[NewPeakIndices,:]
    NewP_fitness = Fitness_Cal(sample, NewP, stdData, gamma)
    # NewP = np.asarray(NewP)
    # NewP_fitness = np.asarray(NewP_fitness)
    # NewPeakIndices = np.asarray(NewPeakIndices)
    TPC_Indice = Close_Clusters(sample, NewPeakIndices, Dist)
    return NewP, NewP_fitness, TPC_Indice, NewPeakIndices
# Perform the merge among TPCs inside the current data chunk Ct
def CE_InChunk(sample, P, P_fitness, stdData, gamma, Dist, TPC_Indice, PeakIndices):
    while 1:
        HistP = P
        P, P_fitness, TPC_Indice, NewPeakIndices = MergeInChunk(P, P_fitness, sample, gamma, stdData, Dist, TPC_Indice, PeakIndices)
        if np.shape(P)[0] == np.shape(HistP)[0]:
            break
    return P, P_fitness
# Identify samples that are close to the boundary between two neighboring TPCs
def Boundary_Instance(Current, Neighbor, Dist, TPC_Indice, sample):
    temp_cluster1 = np.where(TPC_Indice == Current)[0]
    temp_cluster2 = np.where(TPC_Indice == Neighbor)[0]
    temp = np.concatenate([temp_cluster1, temp_cluster2])

    Dc = Dist[Current][Neighbor]
    Dd = []
    for i in range(len(temp)):
        D1 = Dist[Current][temp[i]]
        D2 = Dist[Neighbor][temp[i]]
        Dd.append(abs(D1 - D2))
    if len(Dd) <= 1:
        BD = sample[Current][:]
    else:
        CI = np.argmin(Dd)
        BD = sample[temp[CI]][:]
    return BD
# Assign samples to its nearest cluster centers and obtain their cluster indices
def AssigntoPeaks(pop,pop_index,P,P_I,marked,radius,Dist):
    temp = []
    [N,L] = np.shape(pop)
    for i in range(N):
        distance = Dist[i,P_I]
        if not np.any(np.isin(marked, pop_index[i])):
            if distance < radius:
                temp.append(pop_index[i])
    indices = temp
    return indices
# Secondary merge between clusters doscover in Ct and clusters in the cluster summary
def MegrewithExisting(P, P_Summary, normal_summary, fault_summary, sample, stdData, gamma, PreStd):
    dim = np.shape(P)[1]
    histnormal= normal_summary[:,:dim]
    histnoT = normal_summary[:,dim + 1]
    histnonum = np.shape(histnormal)[0]
    if np.shape(fault_summary)[0] == 0:
        histfault, histfaT, histfanum= [], [], 0
        histclusters, histclustersT = histnormal, histnoT
    else:
        histfault = fault_summary[:,:dim]
        histfaT = fault_summary[:,dim + 2]
        histfanum = np.shape(histfault)[0]
        histclusters = np.concatenate([histnormal,histfault])
        histclustersT = np.concatenate([histnoT,histfaT])
    concate_clusters = np.concatenate([P, histclusters])
    concate_dist = squareform(pdist(concate_clusters))
    [min_dist, cluster_indices] = Cluster_Assign(sample, P)
    RPF = Fitness_Cal(sample, concate_clusters, stdData, gamma)
    PF = fitness_update(P_Summary, concate_clusters, RPF, PreStd, gamma, stdData)
    merge_nei = []
    merge_nornei, merge_falnei = [], []
    remain1, remain2 = [], []
    mergedno, mnofit, mergedfa, mfafit, novel, nofit, remainno, rnofit, rno_T, remainfa, rfafit, rfa_T = [],[],[],[],[],[],[],[],[],[],[],[]
    for i in range(np.shape(P)[0]):
        merge_n, merge_f = True, True
        Current = P[i, :]
        Current = Current.reshape(1, dim)
        NeighborDist = concate_dist[i][-np.shape(histclusters)[0]:]
        Neighbor_Index = np.argmin(NeighborDist)
        if not np.any(np.isin(merge_nei, Neighbor_Index)):
            # Merge check with historical normal clusters
            if Neighbor_Index < (histnonum):
                neighbornormal = histclusters[Neighbor_Index][:]
                neighbornormal = neighbornormal.reshape(1, dim)
                current_cluster = np.where(cluster_indices == i)[0]
                temp_concatenate = np.concatenate([Current, neighbornormal])
                temp_concatenate = np.concatenate([temp_concatenate, sample[current_cluster][:]])
                tempconcate_dist = squareform(pdist(temp_concatenate))
                dist_diff = tempconcate_dist[0][2:] - tempconcate_dist[1][2:]
                dist_diff = np.absolute(dist_diff)
                boundary_sample = (Current + neighbornormal) / 2
                boundary_sample = boundary_sample.reshape(1, dim)
                R_fitness = Fitness_Cal(sample, boundary_sample, stdData, gamma)
                boundary_fitness = fitness_update(P_Summary, boundary_sample, R_fitness, PreStd, gamma, stdData)
                current_fit = PF[i]
                neighbor_fit = PF[np.shape(P)[0] + Neighbor_Index]
                if boundary_fitness < 0.95 * min(current_fit, neighbor_fit):
                    merge_n = False
                else:
                    merge_n = True
                if merge_n:
                    merge_nei.append(Neighbor_Index)
                    merge_nornei.append(Neighbor_Index)
                    if neighbor_fit <= current_fit:
                        mergedno.append(Current)
                        mnofit.append(current_fit)
                    else:
                        mergedno.append(neighbornormal)
                        mnofit.append(neighbor_fit)
                else:
                    novel.append(Current)
                    nofit.append(current_fit)
            else:
                neighborfault = histclusters[Neighbor_Index][:]
                neighborfault = neighborfault.reshape(1, dim)
                current_cluster = np.where(cluster_indices == i)[0]
                temp_concatenate = np.concatenate([Current, neighborfault])
                temp_concatenate = np.concatenate([temp_concatenate, sample[current_cluster][:]])
                tempconcate_dist = squareform(pdist(temp_concatenate))
                dist_diff = tempconcate_dist[0][2:] - tempconcate_dist[1][2:]
                dist_diff = np.absolute(dist_diff)
                boundary_sample = (Current + neighborfault) / 2
                boundary_sample = boundary_sample.reshape(1, dim)
                R_fitness = Fitness_Cal(sample, boundary_sample, stdData, gamma)
                boundary_fitness = fitness_update(P_Summary, boundary_sample, R_fitness, PreStd, gamma, stdData)
                current_fit = PF[i]
                neighbor_fit = PF[np.shape(P)[0] + Neighbor_Index]
                if boundary_fitness < 0.95* min(current_fit, neighbor_fit):
                    merge_f = False
                if merge_f:
                    merge_nei.append(Neighbor_Index)
                    merge_falnei.append(Neighbor_Index-histnonum)
                    if neighbor_fit <= current_fit:
                        mergedfa.append(Current)
                        mfafit.append(current_fit)
                    else:
                        mergedfa.append(neighborfault)
                        mfafit.append(neighbor_fit)
                else:
                    novel.append(Current)
                    nofit.append(current_fit)
        else:
            novel.append(Current)
            nofit.append(current_fit)
    for j in range(np.shape(P_Summary)[0]):
        if j not in merge_nei:
            if j < histnonum:
                remainno.append(histclusters[j][:])
                rnofit.append(PF[j + np.shape(P)[0]])
                rno_T.append(histclustersT[j])
                remain1.append(j)
            else:
                remainfa.append(histclusters[j][:])
                rfafit.append(PF[j + np.shape(P)[0]])
                rfa_T.append(histclustersT[j])
                remain2.append(j-histnonum)

    mergedno = np.reshape(mergedno, (np.shape(mergedno)[0], dim))
    mergedfa = np.reshape(mergedfa, (np.shape(mergedfa)[0], dim))
    novel= np.reshape(novel, (np.shape(novel)[0], dim))
    remainno = np.reshape(remainno, (np.shape(remainno)[0], dim))
    remainfa = np.reshape(remainfa, (np.shape(remainfa)[0], dim))
    rno_T = np.array(rno_T)
    rfa_T = np.array(rfa_T)
    return mergedno, mnofit, mergedfa, mfafit, novel, nofit, remainno, rnofit, rno_T, remainfa, rfafit, rfa_T, \
           merge_nornei, merge_falnei, remain1, remain2
# Extarct and update the normal cluster summary
def Normal_SummaryUpdate(current_normal, normal_f, normal_summary, sample, normal_time, normal_indices):
    dim = np.shape(sample)[1]
    num_nor = np.shape(current_normal)[0]
    normal_sample = sample[normal_indices,:]
    # Update the radius for normal clusters
    normal_Rp = AverageDist(current_normal, normal_summary, normal_sample, dim)
    normal_Rp = np.reshape(normal_Rp, (num_nor, 1))
    normal_T = np.reshape(normal_time, (num_nor, 1))
    normal_F = np.reshape(normal_f, (num_nor, 1))
    PCluster1 = np.concatenate([current_normal, normal_F], axis=1)
    PCluster1 = np.concatenate([PCluster1, normal_Rp], axis=1)
    PCluster1 = np.concatenate([PCluster1, normal_T], axis=1)
    return PCluster1

# Extarct and update the fault cluster summary
def Fault_SummaryUpdate(current_fault, fault_f, fault_summary, sample, fault_time, fault_indices):
    dim = np.shape(sample)[1]
    num_fau = np.shape(current_fault)[0]
    if num_fau == 0:
        return fault_summary
    fault_sample = sample[fault_indices, :]
    # Update the radius for normal clusters
    fault_Rp = AverageDist(current_fault, fault_summary, fault_sample, dim)
    fault_Rp = np.reshape(fault_Rp, (num_fau, 1))
    fault_T = np.reshape(fault_time, (num_fau, 1))
    fault_F = np.reshape(fault_f, (num_fau, 1))
    PCluster2 = np.concatenate([current_fault, fault_F], axis=1)
    PCluster2 = np.concatenate([PCluster2, fault_Rp], axis=1)
    PCluster2 = np.concatenate([PCluster2, fault_T], axis=1)
    return PCluster2
# To validate the existence of clusters
def ClusterValidate(cluster, clusterfit,P_Summary):
    if np.shape(P_Summary)[0] <= 1:
        return cluster, clusterfit
    [Nc,dim] = np.shape(cluster)
    hist_fit = P_Summary[:,dim]
    meanhist = np.mean(hist_fit)
    stdhist = np.std(hist_fit)
    v_hist = []
    v_histf = []
    ci = 0
    for i in range(Nc):
        if abs(clusterfit[i]-meanhist) <= 3*stdhist:
            v_hist.append(cluster[i,:])
            v_histf.append(clusterfit[i])
            ci += 1
    cluster = np.reshape(v_hist,(ci,dim))
    clusterfit = v_histf
    return cluster,clusterfit
# Keep the track of the variance of the data stream and the density values of cluster centers
def StoreInf(PF, PFS, PreStd, stdData):
    PreStd.append(stdData)
    PFS.append(PF)
    return PreStd, PFS
# --------------------Cluster Radius Computation and Update--------------------#
def AverageDist(P, P_Summary, sample, dim):
    # Obtain the assignment of clusters
    [distance, indices] = Cluster_Assign(sample, P)
    rad1 = []
    # if the summary of clusters is not empty
    if len(P_Summary) > 0:
        PreP = P_Summary[:, 0:dim]  # Historical Cluster Center vector
        PreR = P_Summary[:, dim + 1]  # Historical Cluster Radius
        dist_PtoPreP = squareform(pdist(np.concatenate([P, PreP])))
        for i in range(np.shape(P)[0]):
            if np.shape(np.where(indices == i))[1] > 1:
                SumD1 = 0
                Count1 = 0
                for j in range(np.shape(sample)[0]):
                    if indices[j] == i:
                        SumD1 += distance[j]
                        Count1 += 1
                rad1.append(SumD1 / (1+Count1))
            else:
                C_d = dist_PtoPreP[i, np.shape(P)[0]:]
                CI = np.argmin(C_d)
                rad1.append(PreR[CI])
    else:
        for i in range(np.shape(P)[0]):
            SumD1 = 0
            Count1 = 0
            for j in range(np.shape(sample)[0]):
                if indices[j] == i:
                    SumD1 += distance[j]
                    Count1 += 1
            rad1.append(SumD1 / (1+Count1))
    return np.asarray(rad1)
# Perform the active learning for novel class instances
def active_labelquery(num_S, P, sample, temp_oracle, ClusterIndice, num_re, num_me):
    FetchIndex, UnlabeledIndex = [], []
    InterDist = squareform(pdist(P))
    novelno, novelfa = [], []
    novelno_label, novelfa_label = [], []
    # If only one cluster is discovered:
    if np.shape(P)[0] == 1:
        tempcluster = np.where(ClusterIndice == num_re)
        d1 = []
        for j in range(len(tempcluster[0])):
            d1.append(np.linalg.norm(sample[tempcluster[0][j], :] - P))
        fetchSize = round(num_S * len(d1) / (1 + np.shape(sample)[0]))+1
        sortIndex1 = np.argsort(d1)
        fet1 = tempcluster[0][sortIndex1[:ceil(fetchSize * 0.5)]]
        fet2 = tempcluster[0][sortIndex1[-ceil(fetchSize * 0.5):]]
        FetchIndex = np.append(FetchIndex, fet1)
        FetchIndex = np.append(FetchIndex, fet2)
        FetchIndex = FetchIndex.astype(int)
        for ui in range(len(temp_oracle)):
            if not np.any(np.isin(FetchIndex, ui)):
                UnlabeledIndex.append(ui)
        UnlabeledIndex = np.asarray(UnlabeledIndex)
        UnlabeledIndex = UnlabeledIndex.astype(int)
        novellabel = temp_oracle[FetchIndex]
        temp_no = np.unique(novellabel)
        if len(temp_no) == 0:
            novelno, novelfa, novelno_label, novelfa_label = [], [], [], []
            return FetchIndex, UnlabeledIndex, novelno, novelfa, novelno_label, novelfa_label
        if num_me == 1:
            novelno, novelfa, novelno_label, novelfa_label = [], [], [], []
        else:
            if len(temp_no) > 1:
                for diff_label in temp_no:
                    if diff_label[0] == 'F':
                        curr_fa = np.mean(sample[FetchIndex[np.where(novellabel == diff_label)[0]]][:], axis=0)
                        curr_fa = np.reshape(curr_fa, (1, np.shape(P)[1]))
                        if len(novelfa) == 0:
                            novelfa, novelfa_label = curr_fa, [diff_label]
                        else:
                            novelfa = np.concatenate([novelfa, curr_fa])
                            novelfa_label.append(diff_label)
                    else:
                        curr_no = np.mean(sample[FetchIndex[np.where(novellabel == diff_label)[0]]][:], axis=0)
                        curr_no = np.reshape(curr_no, (1, np.shape(P)[1]))
                        if len(novelno) == 0:
                            novelno = curr_no
                            novelno_label = [diff_label]
                        else:
                            novelno = np.concatenate([novelno, curr_no])
                            novelno_label.append(diff_label)
            else:
                if temp_no == 'Normal ':
                    novelno, novelfa = P, []
                    novelno_label, novelfa_label = [temp_no], []
                else:
                    novelno, novelfa = [], P
                    novelno_label, novelfa_label = [], [temp_no]
        return FetchIndex, UnlabeledIndex, novelno, novelfa, novelno_label, novelfa_label
    for i in range(np.shape(P)[0]):
        temp_fetch = []
        tempcluster = np.where(ClusterIndice == (i + num_re))
        d1 = []
        for j in range(len(tempcluster[0])):
            d1.append(np.linalg.norm(sample[tempcluster[0][j], :] - P[i, :]))
        fetchSize = round(num_S * len(d1) / (1+np.shape(sample)[0]))+1
        sortIndex1 = np.argsort(d1)
        fet1 = tempcluster[0][sortIndex1[:ceil(fetchSize * 0.5)]]
        fet2 = tempcluster[0][sortIndex1[-ceil(fetchSize * 0.5):]]
        FetchIndex = np.append(FetchIndex, fet1)
        FetchIndex = np.append(FetchIndex, fet2)
        FetchIndex = FetchIndex.astype(int)
        temp_fetch = np.append(temp_fetch, fet1)
        temp_fetch = np.append(temp_fetch, fet2)
        temp_fetch = temp_fetch.astype(int)
        novellabel = temp_oracle[temp_fetch]
        temp_no = np.unique(novellabel)
        if len(temp_no) == 0:
            # novelfa, novelno = [], []
            # novelno_label, novelfa_label = [], []
            continue
        if len(temp_no) > 1:
            for diff_label in temp_no:
                if diff_label[0] == 'F':
                    curr_fa = np.mean(sample[FetchIndex[np.where(novellabel == diff_label)[0]]][:], axis=0)
                    curr_fa = np.reshape(curr_fa, (1, np.shape(P)[1]))
                    if len(novelfa) == 0:
                        novelfa, novelfa_label = curr_fa, [diff_label]
                    else:
                        novelfa = np.concatenate([novelfa, curr_fa])
                        novelfa_label.append(diff_label)
                else:
                    curr_no = np.mean(sample[FetchIndex[np.where(novellabel == diff_label)[0]]][:], axis=0)
                    curr_no = np.reshape(curr_no, (1, np.shape(P)[1]))
                    if len(novelno) == 0:
                        novelno = curr_no
                        novelno_label = [diff_label]
                    else:
                        novelno = np.concatenate([novelno, curr_no])
                        novelno_label.append(diff_label)
        else:
            if temp_no == 'Normal ':
                if len(novelno) == 0:
                    novelno = np.reshape(P[i, :], (1, np.shape(P)[1]))
                    novelno_label = [temp_no[0]]
                else:
                    novelno = np.concatenate([novelno, np.reshape(P[i, :], (1, np.shape(P)[1]))])
                    novelno_label.append(temp_no[0])
            else:
                if len(novelfa) == 0:
                    novelfa = np.reshape(P[i, :], (1, np.shape(P)[1]))
                    novelfa_label = [temp_no[0]]
                else:
                    novelfa = np.concatenate([novelfa, np.reshape(P[i, :], (1, np.shape(P)[1]))])
                    novelfa_label.append(temp_no[0])
    for ui in range(np.shape(sample)[0]):
        if not np.any(np.isin(FetchIndex, ui)):
            UnlabeledIndex.append(ui)
    UnlabeledIndex = np.asarray(UnlabeledIndex)
    UnlabeledIndex = UnlabeledIndex.astype(int)
    return FetchIndex, UnlabeledIndex, novelno, novelfa, novelno_label, novelfa_label
# Propagate queried labels to the remaining unlabeled novel class instances
def Label_Propagation(sample_Fetch, label_Fetch, sample_Unlabeled):
    Y = []
    for x in sample_Unlabeled:
        vote_dist = []
        for y in sample_Fetch:
            vote_dist.append(np.linalg.norm(x - y))
        vote_order = np.argsort(vote_dist)
        vote_index = vote_order[0]
        vote_res = label_Fetch[vote_index]
        if len(vote_order) < 5:
            vote_index = vote_order[0]
            vote_res = label_Fetch[vote_index]
        else:
            vote_index = vote_order[:5]
            voter = label_Fetch[vote_index]
            [vote_l, vote_c] = np.unique(voter, return_counts=True)
            vote_res = vote_l[np.argmax(vote_c)]
        Y.append(vote_res)
    label_Unfetched = np.asarray(Y)
    return label_Unfetched
# Update the clustering model in terms of subcluster information
def UpdateExistModel(clusterindice, normal_clusters, normal_summary, normalsub_info, fault_clusters, fault_summary,
                     faultsub_info, novelnor, novelfal, sample, obtained_label, stdData, gamma):
    dim = np.shape(normal_clusters)[1]
    # For updating the normal subcluster summary
    hist_center1, curr_center1 = normal_summary[:, :dim], normal_clusters
    num_c1 = np.shape(curr_center1)[0]
    for i in range(num_c1):
        current_norind = np.where(clusterindice == i)[0]
        inter_dist1 = []
        if i >= (num_c1 - np.shape(novelnor)[0]):
            break
        for j in range(np.shape(hist_center1)[0]):
            inter_dist1.append(np.linalg.norm(curr_center1[i, :] - hist_center1[j, :]))
        nearest_ind1 = np.argmin(inter_dist1)
        histsub_info1 = normalsub_info[str(nearest_ind1)]
        hist_labels1 = np.asarray(list(histsub_info1.keys()))
        current_labels1 = obtained_label[current_norind]
        current_labelinfo1 = np.unique(current_labels1)
        for l in current_labelinfo1:
            sample_sub1 = np.where(current_labels1 == l)[0]
            subcluster_curr1 = sample[current_norind[sample_sub1], :]
            subcluster_fit1 = Fitness_Cal(subcluster_curr1, subcluster_curr1, stdData, gamma)
            curr_submean1 = subcluster_curr1[np.argmax(subcluster_fit1)]
            curr_submean1 = curr_submean1.reshape(1, dim)
            curr_subfit1 = np.max(subcluster_fit1)
            curr_subnum1 = np.shape(subcluster_curr1)[0]
            if np.any(np.isin(hist_labels1, str(l))):
                hist_sub1 = histsub_info1[str(l)]
                hist_submean1 = hist_sub1[:, :dim]
                hist_subnum1 = hist_sub1[:, dim]
                sub_num1 = hist_subnum1 + curr_subnum1
                hist_submeanf1 = Fitness_Cal(subcluster_curr1, hist_submean1, stdData, gamma)
                if hist_submeanf1 > curr_subfit1:
                    sub_mean1 = hist_submean1
                else:
                    sub_mean1 = curr_submean1
                hist_sub1[:, :dim] = sub_mean1
                hist_sub1[:, dim] = sub_num1
                histsub_info1[str(l)] = hist_sub1
            else:
                new_sub1 = np.zeros((1, dim + 1))
                new_sub1[:, :dim] = curr_submean1
                new_sub1[:, dim] = curr_subnum1
                if curr_subnum1 >= 1:
                    histsub_info1.update({str(l): new_sub1})
                else:
                    histsub_info1[str(l)] = histsub_info1[str(l)]
        normalsub_info[str(nearest_ind1)] = histsub_info1
    # For updating the fault subcluster summary
    if np.shape(fault_summary)[0] == 0 or len(faultsub_info)==0:
        return normalsub_info, faultsub_info
    hist_center2, curr_center2 = fault_summary[:, :dim], fault_clusters
    num_c2 = np.shape(curr_center2)[0]
    for i in range(num_c2):
        if i >= (num_c2 - np.shape(novelfal)[0]):
            break
        current_falind = np.where(clusterindice == (i + num_c1))[0]
        inter_dist2 = []
        for j in range(np.shape(hist_center2)[0]):
            inter_dist2.append(np.linalg.norm(curr_center2[i, :] - hist_center2[j, :]))
        nearest_ind2 = np.argmin(inter_dist2)
        histsub_info2 = faultsub_info[str(nearest_ind2)]
        hist_labels2 = np.asarray(list(histsub_info2.keys()))
        current_labels2 = obtained_label[current_falind]
        current_labelinfo2 = np.unique(current_labels2)
        for l in current_labelinfo2:
            sample_sub2 = np.where(current_labels2 == l)[0]
            subcluster_curr2 = sample[current_falind[sample_sub2], :]
            subcluster_fit2 = Fitness_Cal(subcluster_curr2, subcluster_curr2, stdData, gamma)
            curr_submean2 = subcluster_curr2[np.argmax(subcluster_fit2)]
            curr_submean2 = curr_submean2.reshape(1, dim)
            curr_subfit2 = np.max(subcluster_fit2)
            curr_subnum2 = np.shape(subcluster_curr2)[0]
            if np.any(np.isin(hist_labels2, str(l))):
                hist_sub2 = histsub_info2[str(l)]
                hist_submean2 = hist_sub2[:, :dim]
                hist_subnum2 = hist_sub2[:, dim]
                sub_num2 = hist_subnum2 + curr_subnum2
                hist_submeanf2 = Fitness_Cal(subcluster_curr2, hist_submean2, stdData, gamma)
                if hist_submeanf2 > curr_subfit2:
                    sub_mean2 = hist_submean2
                else:
                    sub_mean2 = curr_submean2
                hist_sub2[:, :dim] = sub_mean2
                hist_sub2[:, dim] = sub_num2
                histsub_info2[str(l)] = hist_sub2
            else:
                new_sub2 = np.zeros((1, dim + 1))
                new_sub2[:, :dim] = curr_submean2
                new_sub2[:, dim] = curr_subnum2
                if curr_subnum2 >= 1:
                    histsub_info2.update({str(l): new_sub2})
                else:
                    histsub_info2 = histsub_info2
        faultsub_info[str(nearest_ind2)] = histsub_info2
    return normalsub_info, faultsub_info
# Create new clustering models for novel clusters
def CreateNewModel(normal_indices1, normal_clusters, normal_summary, normalsub_info, normal_indice, fault_indices1, fault_clusters,
                   fault_summary, faultsub_info, fault_indice,  novelnor, novelfal, novelno_label, novelfa_label, sample,
                   obtained_label, stdData, gamma):
    dim = np.shape(normal_clusters)[1]
    num_nor1, num_fal1 = np.shape(normal_summary)[0], np.shape(fault_summary)[0]
    for i in range(np.shape(novelnor)[0]):
        insert_subinfo1 = {}
        novel_norind = np.where(normal_indices1 == (i + num_nor1))[0]
        if len(novel_norind) == 0:
            insert_info1 = np.zeros((1, dim + 1))
            insert_info1[0, :dim] = novelnor[i,:]
            insert_info1[0, dim] = 1
            # insert_key = novelno_label[i]
            insert_subinfo1.update({'Normal ': insert_info1})
            normalsub_info.update({str(i + num_nor1): insert_subinfo1})
            continue
        novel_norlabels = obtained_label[normal_indices[novel_norind]]
        novel_labelinfo1 = np.unique(novel_norlabels)
        for lnew in novel_labelinfo1:
            insert_info1 = np.zeros((1, dim + 1))
            sample_sub1 = np.where(novel_norlabels == lnew)[0]
            subcluster_novel1 = sample[novel_norind[sample_sub1], :]
            subcluster_fitness1 = Fitness_Cal(subcluster_novel1,subcluster_novel1,stdData,gamma)
            novel_submean1 = subcluster_novel1[np.argmax(subcluster_fitness1, axis=0)]
            novel_subnum1 = np.shape(subcluster_novel1)[0]
            insert_info1[0, :dim] = novel_submean1
            insert_info1[0, dim] = novel_subnum1
            insert_subinfo1.update({str(lnew): insert_info1})
        normalsub_info.update({str(i + num_nor1): insert_subinfo1})
    for j in range(np.shape(novelfal)[0]):
        insert_subinfo2 = {}
        novel_falind = np.where(fault_indices1 == (j + np.shape(normal_clusters)[0] + num_fal1))[0]
        if len(novel_falind) == 0:
            insert_info2 = np.zeros((1, dim + 1))
            insert_info2[0, :dim] = novelfal[j]
            insert_info2[0, dim] = 1
            insert_subinfo2.update({str(novelfa_label[j]): insert_info2})
            faultsub_info.update({str(j + num_fal1): insert_subinfo2})
            continue
        novel_fallabels = obtained_label[fault_indices[novel_falind]]
        novel_labelinfo2 = np.unique(novel_fallabels)
        for lnew in novel_labelinfo2:
            insert_info2 = np.zeros((1, dim + 1))
            sample_sub2 = np.where(novel_fallabels == lnew)[0]
            subcluster_novel2 = sample[novel_falind[sample_sub2], :]
            subcluster_fitness2 = Fitness_Cal(subcluster_novel2,subcluster_novel2,stdData,gamma)
            novel_submean2 = subcluster_novel2[np.argmax(subcluster_fitness2, axis=0)]
            novel_subnum2 = np.shape(subcluster_novel2)[0]
            insert_info2[0, :dim] = novel_submean2
            insert_info2[0, dim] = novel_subnum2
            insert_subinfo2.update({str(lnew): insert_info2})
        faultsub_info.update({str(j + num_fal1): insert_subinfo2})
    return normalsub_info, faultsub_info
# Extract the sub-cluster representatives from the micro-level summary
def Extract_Labels(normalsub_info, faultsub_info, dim):
    ln = 0
    normalcluster_labels = list(normalsub_info.keys())
    faultcluster_labels = list(faultsub_info.keys())
    if len(normalcluster_labels) == 0 and len(faultcluster_labels) == 0:
        represent, represent_label = [], []
    for l in normalcluster_labels:
        temp_clusterinfo1 = normalsub_info[str(l)]
        sublabels1 = list(temp_clusterinfo1.keys())
        for lsub1 in sublabels1:
            content1 = temp_clusterinfo1[str(lsub1)]
            if ln == 0:
                represent = content1[:,:dim]
                represent_label = lsub1
            else:
                represent = np.append(represent,content1[:, :dim])
                represent_label = np.append(represent_label,lsub1)
            ln += 1
    for l in faultcluster_labels:
        temp_clusterinfo2 = faultsub_info[str(l)]
        sublabels2 = list(temp_clusterinfo2.keys())
        for lsub2 in sublabels2:
            content2 = temp_clusterinfo2[str(lsub2)]
            if ln == 0:
                represent = content2[:, :dim]
                represent_label = lsub2
            else:
                represent = np.append(represent,content2[:, :dim])
                represent_label = np.append(represent_label,lsub2)
            ln += 1
    represent = np.reshape(represent,(ln,dim))
    return represent,represent_label
# Identify the least confidence samples from the merged clusters
def leastconfidence(N_active,subreps,sublabelrep, sample, idx_classify):
    N_sub =  np.shape(subreps)[0]
    subrepsclassify = np.concatenate([subreps,sample[idx_classify,:]])
    classify_dist = squareform(pdist(subrepsclassify))
    classify_dist2 = classify_dist[N_sub:,:N_sub]
    sumdist = []
    for i in range(np.shape(classify_dist2)[0]):
        tempsubdist = classify_dist2[i,:]
        sumdist.append(np.sum(tempsubdist))
    sort_sumdist = np.argsort(sumdist)
    return idx_classify[sort_sumdist[:N_active]], idx_classify[sort_sumdist[N_active:]]
# Learning and classifying the incoming data chunk
def learnandclassify(clusters, normalsub_info, faultsub_info, num_re, num_me, num_no, sample, clusteridx, temp_oracle, queryhist, dim):
    num_S = round(label_ratio/100 * len(clusteridx))
    re_cluster = clusters[:num_re, :]
    merged_cluster = clusters[num_re:(num_re + num_me)]
    obtained_label = [None] * np.shape(sample)[0]
    obtained_label = np.asarray(obtained_label)
    idx_toclassifyhard = []
    FetchIndex1 = []
    if num_me > 0 or num_no > 0:
        subcluster_representatives, subcluster_labelrep = Extract_Labels(normalsub_info, faultsub_info, dim)
        FetchIndex1 = []
        if num_re > 0 and num_no == 0  and num_me > 0:
            idx_toclassify = np.where(clusteridx < num_re)[0]
            numf1 = round(num_S * len(idx_toclassify) / len(clusteridx))+1
            # Perform the uncertainty-based sampling for the existing clusters
            if len(idx_toclassify) > 0:
                idx_toclassifyhard, idx_easyclassify = leastconfidence(numf1, subcluster_representatives,
                subcluster_labelrep, sample, idx_toclassify)
                FetchIndex1 = idx_toclassifyhard
            else:
                idx_toclassifyhard, idx_easyclassify = [], []
            idx_remain = np.where(clusteridx >= num_re)[0]
        else:
            idx_easyclassify = []
            idx_remain = np.arange(0, np.shape(sample)[0])
            numf1 = 0
        query_clusters = clusters[num_re:, :]
        sample_remain = sample[idx_remain, :]
        num_f2 = num_S - numf1
        FetchIndex2, UnfetchedIndex, novelno, novelfa, novelno_label, novelfa_label = active_labelquery(num_f2, query_clusters, sample_remain,
        temp_oracle[idx_remain], clusteridx[idx_remain], num_re, num_me)
        FetchIndex = np.concatenate([FetchIndex1, idx_remain[FetchIndex2]])
#        FetchIndex = idx_remain[FetchIndex2]
        FetchIndex = FetchIndex.astype(int)
        sample_Fetch = sample[FetchIndex][:]
        label_Fetch = temp_oracle[FetchIndex]
        queryhist.append(len(label_Fetch))
        # Extract the indices of unlabeled samples
        unclassified_idx = np.append(idx_easyclassify, idx_remain[UnfetchedIndex])
        unclassified_idx = unclassified_idx.astype(int)
        unclassified_sample = sample[unclassified_idx][:]
        if len(normalsub_info) > 0 or len(faultsub_info) > 0:
            combined_reps = np.concatenate([subcluster_representatives, sample_Fetch], axis=0)
            combined_repls = np.append(subcluster_labelrep, label_Fetch)
        else:
            combined_reps = sample_Fetch
            combined_repls = label_Fetch
        propagated_labels = Label_Propagation(combined_reps, combined_repls, unclassified_sample)
        obtained_label[FetchIndex] = label_Fetch
        obtained_label[unclassified_idx] = propagated_labels
    else:
        subcluster_representatives, subcluster_labelrep = Extract_Labels(normalsub_info, faultsub_info, dim)
        unclassified_idx = np.arange(0, np.shape(sample)[0])
        unclassified_sample = sample[unclassified_idx][:]
        combined_reps = subcluster_representatives
        combined_repls = subcluster_labelrep
        propagated_labels = Label_Propagation(combined_reps, combined_repls, unclassified_sample)
        obtained_label[unclassified_idx] = propagated_labels
        novelno, novelfa = [], []
        novelno_label, novelfa_label = [], []
    return obtained_label, novelno, novelfa, novelno_label, novelfa_label, queryhist,unclassified_idx
# Refine the maintained summary of clusters when faults are not captured by the clustering procedure
def ModelRefinement(normal_summary, normalsub_info, fault_summary, faultsub_info):
    [num_nor, cols] = np.shape(normal_summary)
    dim = cols - 3
    num_fal1, num_fal2 = np.shape(fault_summary)[0], np.shape(fault_summary)[0]
    add1, add2 = 0, 0
    count_del1, count_del2 = 0, 0
    fdel_idx, ndel_idx = [], []
    normal_clusters = normal_summary[:,:dim]
    newnormalsub, newfaultsub = {}, {}
    # In case of empty fault summary
    if num_fal1 == 0:
        fault_clusters = []
    else:
        fault_clusters = fault_summary[:,:dim]
    # Refine on normal cluster summary
    for i in range(num_nor):
        templabelinfor1 = normalsub_info[str(i)]
        templabelset1 = np.unique(list(templabelinfor1.keys()))
        if len(templabelset1) == 1:
            if templabelset1 != 'Normal ':
                templabelinfo1 = templabelinfor1[str(templabelset1[0])]
                normal_clusters = np.delete(normal_clusters, i-count_del1, axis=0)
                del normalsub_info[str(i)]
                if count_del1 == 0:
                    ndel_idx = [i]
                else:
                    ndel_idx.append(i)
                count_del1 += 1
                if num_fal1 == 0:
                    fault_clusters = np.reshape(templabelinfo1[0][:dim+1], (1, dim+1))
                else:
                    fault_clusters = np.concatenate([fault_clusters, np.reshape(templabelinfo1[0][:dim], (1, dim))])
                faultsub_info[str(num_fal2+add1)] = {str(templabelset1[0]):templabelinfo1}
                add1 += 1
                num_fal1 += 1
        else:
            for k1 in range(len(templabelset1)):
                l1 = templabelset1[k1]
                templabelinfol = templabelinfor1[str(l1)]
                if l1 != 'Normal ':
                    if num_fal1 == 0:
                        fault_clusters = np.reshape(templabelinfol[0][:dim], (1, dim))
                    else:
                        fault_clusters = np.concatenate([fault_clusters, np.reshape(templabelinfol[0][:dim], (1, dim))])
                    faultsub_info[str(num_fal2 + add1)] = {str(l1): templabelinfor1[str(l1)]}
                    add1 += 1
                    num_fal1 += 1
                    templabelinfor1.pop(str(l1))
                else:
                    normal_clusters[i-count_del1, :] = templabelinfol[0][:dim]
            # normalsub_info[str(i)] = templabelinfor1
    # Update the number of normal clusters after the refinement of normal clusters
    num_nor -= count_del1
    for j in range(num_fal1):
        templabelinfor2 = faultsub_info[str(j)]
        templabelset2 = np.unique(list(templabelinfor2.keys()))
        if len(templabelset2) == 1:
            if templabelset2 == 'Normal ':
                templabelinfo2 = templabelinfor2[str(templabelset2[0])]
                if count_del2 == 0:
                    fdel_idx = [j]
                else:
                    fdel_idx.append(j)
                normal_clusters = np.concatenate([normal_clusters,np.reshape(templabelinfo2[0][:dim], (1, dim))])
                normalsub_info[str(num_nor+add2)] = {str(templabelset2[0]):np.reshape(templabelinfo2[0][:dim+1], (1, dim+1))}
                fault_clusters = np.delete(fault_clusters, (j - count_del2), axis=0)
                del faultsub_info[str(j)]
                # faultsub_info.pop(str(j))
                add2 += 1
                count_del2 += 1
        else:
            for k2 in range(len(templabelset2)):
                l2 = templabelset2[k2]
                templabelinfo2 = templabelinfor2[str(l2)]
                if l2 == 'Normal ':
                    normalsub_info[str(num_nor + add2)] = {str(l2):templabelinfor2[str(l2)]}
                    normal_clusters = np.concatenate([normal_clusters, np.reshape(templabelinfo2[0][:dim], (1, dim))])
                    add2 += 1
                    templabelinfor2.pop(str(l2))
                else:
                    fault_clusters[j-count_del2, :] = templabelinfo2[0][:dim]
            # faultsub_info[str(j)] = templabelinfor2
    # Update the keys of the fault sub-cluster summaries
    newfkeysnum = np.arange(np.shape(fault_clusters)[0])
    # fkeys = list(faultsub_info.keys())
    newfkeys = []
    for k1 in newfkeysnum:
        # if len()
        newfkeys.append(str(k1))
    faultsub_info = dict(zip(newfkeys, faultsub_info.values()))
    # Update the keys of the normal sub-cluster summaries
    newnkeysnum = np.arange(np.shape(normal_clusters)[0])
    newnkeys = []
    for k2 in newnkeysnum:
        newnkeys.append(str(k2))
    normalsub_info = dict(zip(newnkeys, normalsub_info.values()))
    return normal_clusters, normalsub_info, fault_clusters, faultsub_info, ndel_idx, fdel_idx
def feature_associate(combined_reps, combined_repls, fea_relinfo):
    com_faultidx = np.where(combined_repls != 'Normal ')[0]
    com_normalidx = np.where(combined_repls == 'Normal ')[0]
    fault_sample, fault_label = combined_reps[com_faultidx][:], combined_repls[com_faultidx]
    normal_sample, normal_label = combined_reps[com_normalidx][:], combined_repls[com_normalidx]
    com_falset = np.unique(fault_label)
    identified_keys = list(fea_relinfo.keys())
    if len(com_falset) == 0:
        return fea_relinfo
    fea_relinfo1 = {}
    for i in range(len(com_falset)):
        if com_falset[i] == '[]':
            continue
        curr_idx = np.where(fault_label == com_falset[i])[0]
        no_idx = np.where(fault_label != com_falset[i])[0]
        check_sample = np.concatenate([fault_sample[curr_idx,:], normal_sample])
        check_sample = np.concatenate([check_sample, fault_sample[no_idx,:]])
        check_label = np.concatenate([fault_label[curr_idx], normal_label])
        check_label = np.concatenate([check_label, np.asarray(['Normal ']*len(no_idx))])
#        curr_fel = fisher_score(check_sample, check_label)
        curr_fel = mutual_info_classif(check_sample, check_label)
        curr_fel = curr_fel / (1+np.sum(curr_fel))
#        curr_rank = np.argsort(curr_fel)[::-1]
        fea_relinfo1[str(com_falset[i])] = selection_criteria(curr_fel)
#        fea_relinfo1[str(com_falset[i])] = curr_fel
#        fea_relinfo1[str(com_falset[i])] = curr_rank[:3]
    fea_relinfo = fea_relinfo1
    return fea_relinfo
def selection_criteria(fea_rel):
    sensor_id = np.asarray(['gps','gps','imu','imu','imu','imu','compass'])
    unique_id = np.unique(sensor_id)
    aggregate_rel = []
    for idx in unique_id:
        associated_att = np.where(sensor_id == idx)[0]
        aggregate_rel.append(np.max(fea_rel[associated_att]))
    mu_aggrel = np.mean(aggregate_rel)
    sigma_aggrel = np.std(aggregate_rel)
    lb = mu_aggrel - sigma_aggrel
    predicted_idx = np.argmax(aggregate_rel)
#    predicted_idx = np.where(aggregate_rel >= lb)[0]
#    return unique_id[predicted_idx]
    if np.sum(aggregate_rel) == 0:
        return aggregate_rel / (1+np.sum(aggregate_rel))
    else:
        return aggregate_rel / np.sum(aggregate_rel)
def extract_faultinfo(faultsub_info, dim):
    if len(faultsub_info) == 0:
        return 0
    fc_size = []
    count = 0
    for idx in faultsub_info.keys():
        count += 1
        faultcinfo = faultsub_info[idx]
        for subidx in faultcinfo.keys():
            target_info = faultcinfo[subidx]
            fc_size.append(target_info[0,dim]) 
    return np.mean(fc_size)
# ---------------------------Main Function-------------------------#
if __name__ == '__main__':
    [data, label, attributes] = Input()
    start = time.time()
    # data = time_sequence(data,3)
    dim = np.shape(data)[1]
    [BufferSize, normal_summary, fault_summary, T, PFS, PreStd] = ParamSpe(data)
    mu = None
    std = None
    histmu = None
    histstd = None
    AccHist = []
    BAcc1_Hist = []
    F1Hist = []
    P1Hist = []
    R1Hist = []
    Tc = []
    normalsub_info = {}
    faultsub_info = {}
    gammaHist = []
    PFS = []
    PreMu = []
    acc_predlabel = []
    acc_truelabel = []
    acc_faultclasses = []
    AccFaultLabel = []
    queryhist = []
    no_faulttrack = []
    mergedfa = []
    mergedno = []
    faultcount = 0
    truecount = 0
    falsecount = 0
    misscount = 0
    ndel_idx, fdel_idx = [], []
    fea_relinfo = {}
    steps = 0
    labeling_budget = 0
    label_ratio = 10
    for t in range(T):
        # Passing the data stream in a sequence of data chunks
        if t == 0:
            sample = data[t * BufferSize:(t + 1) * BufferSize, :]
            temp_oracle = label[t * BufferSize:(t + 1) * BufferSize]
        elif t < T - 1:
            sample = data[t * BufferSize-steps:(t + 1) * BufferSize-steps, :]
            temp_oracle = label[t * BufferSize-steps:(t + 1) * BufferSize-steps]
        else:
            sample = data[t * BufferSize-steps:np.shape(data)[0]]
            temp_oracle = label[t * BufferSize-steps:np.shape(data)[0]]
        predict_fault = False
        true_fault = False
        # Keep track of a stack of the processed data chunks
        if t == 0:
            AccSample, AccLabel = sample, temp_oracle
        else:
            AccSample = np.concatenate([AccSample, sample])
            AccLabel = np.concatenate([AccLabel, temp_oracle])
        mu = np.mean(sample, axis=0)
        std = np.mean(sample, axis=0)
        # Initiailize the population set and extract its characteristics
        [stdData, pop_index, pop, radius, PreMu, PreStd] = PopInitial(sample, PreMu, PreStd, BufferSize)
        # Initialize the fitness vector
        fitness = np.zeros((len(pop_index), 1))
        # Initialize the indices vector
        indices = np.zeros((len(pop_index), 1))
        # Distance Calculations
        Dist = Distance_Cal(sample)

        # Parameter estimation for Gaussian kernel density function
        if PreStd:
            if np.shape(fault_summary)[0] == 0:
                P_summary = normal_summary
            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
            gam = gamma
            gamma = DCCA(sample, stdData, P_Summary, gam, dim)
        else:
            gamma = CCA(sample, stdData, Dist)
        gammaHist.append(gamma)
        # Fitness calculation
        fitness = Fitness_Cal(sample, pop, stdData, gamma)
        fitness = np.array(fitness)
        P, P_fitness, TPC_Indice, PeakIndices = TPC_Search(Dist, pop_index, pop, radius, fitness)
        P, P_fitness = CE_InChunk(sample, P, P_fitness, stdData, gamma, Dist, TPC_Indice, PeakIndices)

        #        P, P_fitness = NovelClusterValidate(P, P_fitness, P_Summary)
        if t == 0:
            PF = np.asarray(P_fitness)
            #            if np.shape(P)[0] == 1:
            #                P, PF = forcetosplit(P, sample, fitness, PF, Dist)
            TC = np.zeros(np.shape(P)[0])
            # Set the unchanged and merged cluster sets as empty, all discovered clusters are novel
            remain_cluster = []
            merged_cluster = []
            novel_cluster = P
            num_re, num_me, num_no = 0, 0, np.shape(P)[0]
            clusters = novel_cluster
        else:
            if np.shape(fault_summary)[0] == 0:
                P_Summary = normal_summary
            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
            P_fitness = fitness_update(P_Summary, P, P_fitness, PreStd, gamma, stdData)

            [mergedno, mnofit, mergedfa, mfafit, novel, nofit, remainno, rnofit, rnoT, remainfa, rfafit, rfa_T,
             merge_nei1, merge_nei2, remain1, remain2] = MegrewithExisting(P, P_Summary, normal_summary, fault_summary,
                                                                           sample, stdData, gamma, PreStd)
            # Concatenate all unchanged clusters and their densities
            remain_cluster = np.concatenate([remainno, remainfa])
            remain_fit = np.concatenate([rnofit, rfafit])
            # Concatenate all merged clusters and their densities
            merged_cluster = np.concatenate([mergedno, mergedfa])
            merged_fit = np.concatenate([mnofit, mfafit])
            # Validate the occurrence of novel clusters
            novel_cluster, novel_fit = novel, nofit
            # novel_cluster, novel_fit = NovelClusterValidate(merged_cluster, merged_fit, novel, nofit, P_Summary)
            # Extract the time information of all clusters
            remain_T = np.concatenate([rnoT, rfa_T])
            mno_T = t * np.ones(np.shape(mergedno)[0])
            mfa_T = t * np.ones(np.shape(mergedfa)[0])
            merged_T = np.concatenate([mno_T, mfa_T])
            novel_T = t * np.ones(np.shape(novel_cluster)[0])
            # Extract the sizes of unchanged, merged and novel cluster sets
            num_re = np.shape(remain_cluster)[0]
            num_me = np.shape(merged_cluster)[0]
            num_no = np.shape(novel_cluster)[0]
            # Concatenate all clusters in the data stream until time t
            clusters = np.concatenate([remain_cluster, merged_cluster])
            clusters = np.concatenate([clusters, novel_cluster])
        # ------------------Active label querying and classify----------------- #
        [clusterdist, clusteridx] = Cluster_Assign(sample, clusters)
        obtained_label, novelnor, novelfal, novelno_label, novelfa_label, queryhist, unclassified_idx = learnandclassify(clusters, normalsub_info, faultsub_info, num_re, num_me,
                                                              num_no, sample, clusteridx, temp_oracle, queryhist, dim)
        # Check whether some samples are assigned to the historical fault clusters
        if num_re > 0:
            uniquecid = np.unique(clusteridx)
            for uidx in uniquecid:
                ucount = len(np.where(clusteridx == uidx)[0])
                num_rfa = np.shape(remainfa)[0]
                num_rno = np.shape(remainno)[0]
                if uidx < num_re and uidx > num_rno:
                    predict_fault = True
                    break
        if t == 0:
            normal_clusters = novelnor
            normal_T = t * np.ones(np.shape(novelnor)[0])
            fault_clusters = novelfal
            fault_T = t * np.ones(np.shape(novelfal)[0])
        else:
            # Collect the normal cluster information at time t
            merge_nei1 = np.asarray(merge_nei1)
            remain1 = np.asarray(remain1)
            re_order1 = np.concatenate([remain1, merge_nei1])
            re_order1 = re_order1.astype(int)
            re_order1 = np.argsort(re_order1)
            novelnor = np.reshape(novelnor, (np.shape(novelnor)[0], dim))
            normal_clusters = np.concatenate([remainno, mergedno])
            normal_T = np.concatenate([rnoT, mno_T])
            if len(re_order1) > 0:
                normal_clusters = normal_clusters[re_order1][:]
                normal_T = normal_T[re_order1]
            normal_clusters = np.concatenate([normal_clusters, novelnor])
            nno_T = t * np.ones(np.shape(novelnor)[0])
            normal_T = np.concatenate([normal_T, nno_T])
            # Collect the fault cluster information at time t
            merge_nei2 = np.asarray(merge_nei2)
            remain2 = np.asarray(remain2)
            re_order2 = np.concatenate([remain2, merge_nei2])
            re_order2 = re_order2.astype(int)
            re_order2 = np.argsort(re_order2)
            novelfal = np.reshape(novelfal, (np.shape(novelfal)[0], dim))
            fault_clusters = np.concatenate([remainfa, mergedfa])
            fault_T = np.concatenate([rfa_T, mfa_T])
            if len(re_order2) > 0:
                fault_clusters = fault_clusters[re_order2][:]
                fault_T = fault_T[re_order2]
            fault_clusters = np.concatenate([fault_clusters, novelfal])
            nfa_T = t * np.ones(np.shape(novelfal)[0])
            fault_T = np.concatenate([fault_T, nfa_T])
        if np.shape(fault_clusters)[0] == 0:
            current_peaks = normal_clusters
        else:
            current_peaks = np.concatenate([normal_clusters, fault_clusters])
        [MinDist, ClusterIndice] = Cluster_Assign(sample, current_peaks)
        normal_indices = np.where(ClusterIndice < np.shape(normal_clusters)[0])[0]
        fault_indices = np.where(ClusterIndice >= np.shape(normal_clusters)[0])[0]
        num_no = np.shape(novelfal)[0]+np.shape(novelnor)[0]
#        if t == 4:
#            print('Error Check!!!')
        # -------------------Sub-cluster  structure exploration----------------- #
        if t == 0:
            # Explore sub-clusters inside normal clusters
            for n_idx in range(np.shape(novelnor)[0]):
                sample_ind1 = np.where(ClusterIndice == n_idx)[0]
                #                if len(sample_ind1) == 0:
                #                    continue
                pre_label1 = obtained_label[sample_ind1]
                tempcluster_labels1 = np.unique(obtained_label[sample_ind1])
                if len(tempcluster_labels1) > 1:
                    inner_info1 = {}
                    for sublabel1 in tempcluster_labels1:
                        sub_info1 = np.empty((1, dim + 1))
                        subcluster_ind1 = sample_ind1[np.where(pre_label1 == sublabel1)[0]]
                        subcluster1 = sample[subcluster_ind1, :]
                        sub_meanfit1 = Fitness_Cal(subcluster1, subcluster1, stdData, gamma)
                        sub_mean1 = subcluster1[np.argmax(sub_meanfit1), :]
                        sub_num1 = len(subcluster_ind1)
                        sub_info1[0][:dim] = sub_mean1
                        sub_info1[0][dim] = sub_num1
                        if sub_num1 > 10:
                            inner_info1.update({str(sublabel1): sub_info1})
                else:
                    sub_info1 = np.empty((1, dim + 1))
                    subcluster1 = sample[sample_ind1, :]
                    sub_meanfit1 = Fitness_Cal(subcluster1, subcluster1, stdData, gamma)
                    sub_mean1 = subcluster1[np.argmax(sub_meanfit1), :]
                    sub_num1 = len(sample_ind1)
                    sub_info1[0][:dim] = sub_mean1
                    sub_info1[0][dim] = sub_num1
                    inner_info1 = {str(tempcluster_labels1[0]): sub_info1}
                normalsub_info.update({str(n_idx): inner_info1})
            # Explore sub-clusters inside fault clusters
            for f_idx in range(np.shape(novelfal)[0]):
                sample_ind2 = np.where(ClusterIndice == f_idx + (np.shape(novelnor)[0]))[0]
                pre_label2 = obtained_label[sample_ind2]
                tempcluster_labels2 = np.unique(obtained_label[sample_ind2])
                if len(tempcluster_labels2) > 1:
                    inner_info2 = {}
                    for sublabel2 in tempcluster_labels2:
                        sub_info2 = np.empty((1, dim + 1))
                        subcluster_ind2 = sample_ind2[np.where(pre_label2 == sublabel2)[0]]
                        subcluster2 = sample[subcluster_ind2, :]
                        sub_meanfit2 = Fitness_Cal(subcluster2, subcluster2, stdData, gamma)
                        sub_mean2 = subcluster2[np.argmax(sub_meanfit2), :]
                        sub_num2 = len(subcluster_ind2)
                        sub_info1[0][:dim] = sub_mean2
                        sub_info1[0][dim] = sub_num2
                        if sub_num2 > 5:
                            inner_info2.update({str(sublabel2): sub_info2})
                else:
                    sub_info2 = np.empty((1, dim + 1))
                    subcluster2 = sample[sample_ind2, :]
                    sub_meanfit2 = Fitness_Cal(subcluster2, subcluster2, stdData, gamma)
                    sub_mean2 = subcluster2[np.argmax(sub_meanfit2), :]
                    sub_num2 = len(sample_ind2)
                    sub_info2[0][:dim] = sub_mean2
                    sub_info2[0][dim] = sub_num2
                    inner_info2 = {str(tempcluster_labels2[0]): sub_info2}
                faultsub_info.update({str(f_idx): inner_info2})
        else:
            if num_no == 0:
                normalsub_info, faultsub_info = UpdateExistModel(ClusterIndice, normal_clusters, normal_summary,
                                                                 normalsub_info,
                                                                 fault_clusters, fault_summary, faultsub_info, novelnor,
                                                                 novelfal, sample, obtained_label, stdData, gamma)
            else:
                normalsub_info, faultsub_info = UpdateExistModel(ClusterIndice, normal_clusters, normal_summary,
                                                                 normalsub_info, fault_clusters, fault_summary,
                                                                 faultsub_info, novelnor, novelfal, sample,
                                                                 obtained_label, stdData, gamma)
                normalsub_info, faultsub_info = CreateNewModel(ClusterIndice[normal_indices], normal_clusters,
                                                               normal_summary, normalsub_info, normal_indices,
                                                               ClusterIndice[fault_indices], fault_clusters,
                                                               fault_summary, faultsub_info, fault_indices, novelnor, novelfal,
                                                               novelno_label, novelfa_label, sample, obtained_label, stdData, gamma)
                # num_nor2, num_fal2 = np.shape(normal_clusters)[0], np.shape(fault_clusters)[0]
                # if len(invalid_nor) > 0:
                #     normal_clusters = np.delete(normal_clusters, invalid_nor , axis = 0)
                #     normal_T = np.delete(normal_T, invalid_nor, axis = 0)
                # if len(invalid_fal) > 0:
                #     fault_clusters = np.delete(fault_clusters, num_fal2 + invalid_fal - 1, axis=0)
                #     fault_T = np.delete(fault_T, num_fal2 - invalid_fal - 1, axis=0)
        # Update the normal and fault cluster summary
        if t == 0:
            normal_fitness = Fitness_Cal(sample, normal_clusters, stdData, gamma)
            fault_fitness = Fitness_Cal(sample, fault_clusters, stdData, gamma)
            if np.shape(fault_summary)[0] == 0:
                P_Summary = normal_summary
            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
        else:
            if np.shape(fault_summary)[0] == 0:
                P_Summary = normal_summary
            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
            rawnormal_fitness = Fitness_Cal(sample, normal_clusters, stdData, gamma)
            normal_fitness = fitness_update(P_Summary, normal_clusters, rawnormal_fitness, PreStd, gamma, stdData)
            rawfault_fitness = Fitness_Cal(sample, fault_clusters, stdData, gamma)
            fault_fitness = fitness_update(P_Summary, fault_clusters, rawfault_fitness, PreStd, gamma, stdData)
        normal_summary = Normal_SummaryUpdate(normal_clusters, normal_fitness, normal_summary, sample, normal_T,
                                              normal_indices)
        fault_summary = Fault_SummaryUpdate(fault_clusters, fault_fitness, fault_summary, sample, fault_T,
                                            fault_indices)
        num_noro, num_falo = np.shape(normal_clusters)[0], np.shape(fault_clusters)[0]
        if np.shape(normal_summary)[0] > 0 or np.shape(fault_summary)[0] > 0:
            normal_clusters, normalsub_info, fault_clusters, faultsub_info, ndel_idx, fdel_idx = ModelRefinement(
                normal_summary, normalsub_info,
                fault_summary, faultsub_info)
        if t == 0:
            normal_fitness = Fitness_Cal(sample, normal_clusters, stdData, gamma)
            fault_fitness = Fitness_Cal(sample, fault_clusters, stdData, gamma)
            if np.shape(fault_summary)[0] == 0:
                P_Summary = normal_summary
            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
        else:
            rawnormal_fitness = Fitness_Cal(sample, normal_clusters, stdData, gamma)
            normal_fitness = fitness_update(P_Summary, normal_clusters, rawnormal_fitness, PreStd, gamma, stdData)
            if np.shape(fault_summary)[0] == 0:
                P_Summary = normal_summary

            else:
                P_Summary = np.concatenate([normal_summary, fault_summary])
            if np.shape(fault_clusters)[0] == 0:
                rawfault_fitness = []
                fault_fitness = []
            else:
                rawfault_fitness = Fitness_Cal(sample, fault_clusters, stdData, gamma)
                fault_fitness = fitness_update(P_Summary, fault_clusters, rawfault_fitness, PreStd, gamma, stdData)
        num_wrong1 = 0
        num_wrong2 = 0
        if len(fdel_idx) > 0:
            fault_T = np.delete(fault_T, fdel_idx)
        if len(ndel_idx) > 0:
            normal_T = np.delete(normal_T, ndel_idx)

        if len(fault_T) < np.shape(fault_clusters)[0]:
            num_wrong1 = np.shape(fault_clusters)[0] - num_falo
            add_FT = np.shape(fault_clusters)[0] - np.shape(fault_T)[0]
            fault_T = np.concatenate([fault_T, t * np.ones(add_FT)])
        #            print("Novel fault appear!!!")
        if len(normal_T) < np.shape(normal_clusters)[0]:
            num_wrong2 = np.shape(normal_clusters)[0] - num_noro
            add_NT = np.shape(normal_clusters)[0] - np.shape(normal_T)[0]
            normal_T = np.concatenate([normal_T, t * np.ones(add_NT)])

        normal_summary = Normal_SummaryUpdate(normal_clusters, normal_fitness, normal_summary, sample, normal_T,
                                              normal_indices)
        fault_summary = Fault_SummaryUpdate(fault_clusters, fault_fitness, fault_summary, sample, fault_T,
                                            fault_indices)
        print("-------------Fault Classification of Data Stream at time T=" + str(t) + "-------------")

        truelabel_set = np.unique(temp_oracle)

        if len(truelabel_set) == 1:
            if truelabel_set != 'Normal ':
                true_fault = True
        else:
            for tls in truelabel_set:
                if tls != 'Normal ':
                    true_fault = True
                    break
        if true_fault:
            truecount += 1
            print('Fault happens!!!')

        predlabel = obtained_label
        predlabelset = np.unique(predlabel[unclassified_idx])
        threshold = extract_faultinfo(faultsub_info, dim)
        threshold *= 0.3
        if len(predlabelset) == 1:
            if predlabelset[0] != 'Normal ' and len(np.where(predlabel[unclassified_idx]!='Normal ')[0])>=threshold:
                predict_fault = True
            else:
                predict_fault = False
        else:
            for tl in predlabelset:
                if tl != 'Normal ' and len(np.where(predlabel[unclassified_idx]!='Normal ')[0])>=threshold:
                    predict_fault = True
                    break
                else:
                    predict_fault = False
        fea_relinfo = fea_relinfo
        if predict_fault and true_fault:
            fault_idx = unclassified_idx[np.where(temp_oracle[unclassified_idx] != 'Normal ')[0]]
            faultcount += 1
            subcluster_representatives, subcluster_labelrep = Extract_Labels(normalsub_info, faultsub_info, dim)
#            consider_sample = np.concatenate([sample[unclassified_idx], subcluster_representatives])
#            consider_label = np.concatenate([obtained_label[unclassified_idx], subcluster_labelrep])
            consider_sample = np.concatenate([sample, subcluster_representatives])
            consider_label = np.concatenate([obtained_label, subcluster_labelrep])
            fea_relinfo = feature_associate(consider_sample, consider_label, fea_relinfo)
            if len(fault_idx) == 0:
                continue
            else:
                AccHist.append(accuracy_score(temp_oracle[fault_idx], predlabel[fault_idx]) * 100)
                F1Hist.append(f1_score(temp_oracle[fault_idx], predlabel[fault_idx], average='macro') * 100)
                P1Hist.append(precision_score(temp_oracle[fault_idx], predlabel[fault_idx], average='weighted') * 100)
                R1Hist.append(recall_score(temp_oracle[fault_idx], predlabel[fault_idx], average='weighted') * 100)
            print("Faults are correctly detected!!!")
        elif predict_fault and not true_fault:
            falsecount += 1
            print('False faults are detected!!!')
        elif true_fault and not predict_fault:
            misscount += 1
            print('Fail to detect fault!!!')
        else:
#            AccHist.append(0 * 100)
#            F1Hist.append(0 * 100)
#            P1Hist.append(0 * 100)
#            R1Hist.append(0 * 100)
            print('Everything works fine!!!')
        PF = np.concatenate([normal_fitness, fault_fitness])
        PreStd, PFS = StoreInf(PF, PFS, PreStd, stdData)
        acc_predlabel = np.concatenate([acc_predlabel, obtained_label])
        PreStd, PFS = StoreInf(PF, PFS, PreStd, stdData)
        novelnor, novelfal = [], []
        merged_cluster, merged_fit, novel_cluster, novel_fit, re_cluster, re_fit = [], [], [], [], [], []
        mergedfa = []
        mergedno = []
        remainfa = []
        remainno = []
        re_T = []
        fault_T = []
        normal_T = []
    non_zeros = np.nonzero(AccHist)[0]
    AccHist = np.asarray(AccHist)
    F1Hist = np.asarray(F1Hist)
    P1Hist = np.asarray(P1Hist)
    R1Hist = np.asarray(R1Hist)

    normalcounts = T - truecount
#    for key in fea_relinfo.keys():
#        fea_relinfo[key] = attributes[fea_relinfo[key]]
    print("----------Evaluation results on the overall data streams----------")

    if normalcounts > 0:
        TPR = (faultcount) / (faultcount + misscount) * 100
        PPV = 100 - (falsecount) / (falsecount + faultcount) * 100
        print("-------------------Fault detection performance--------------------")
        print("True positive rate: ", TPR)
        print("Precision: ", PPV)
    else:
        TPR = (faultcount) / (faultcount + misscount) * 100
        print("-------------------Fault detection performance--------------------")
        print("True positive rate: ", TPR)
        print("Precision : NA ")
    print("----------------Fault Diagnosis Performance---------------")
    print("The mean of classification accuracy: ", np.mean(AccHist[non_zeros]))
    print("The mean of F1-macro score: ", np.mean(F1Hist[non_zeros]))
    print("Associated features with fault classes: ", fea_relinfo)
    end = time.time()
    print("Running time: ", (end - start))
    print("----------------Labeling Budget-----------------")
    print("Label ratio: ", np.sum(queryhist)/np.shape(data)[0]*100)
    label_set1 = np.unique(AccLabel)
    label_set2 = np.unique(acc_predlabel)
    count1, count2 = 1, 1
    for l1 in label_set2:
        idx1 = np.where(AccLabel==l1)[0]
        if count1 == 5: # 10
            count1=0
        AccLabel[idx1] = count1
        count1 += 1
    for l2 in label_set2:
        idx2 = np.where(acc_predlabel==l2)[0]
        if count2 == 5:# 10
            count2=0
        acc_predlabel[idx2] = count2
        count2 += 1
    t = np.arange(len(AccLabel))
    plt.figure(1)
    plt.plot(t,AccLabel,'ro-', label='ground truth')   
    plt.plot(t,acc_predlabel,'g--', label='predicted')   
    plt.xlabel('Sample id')
    plt.ylabel('Class label')
    plt.legend()
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.savefig('results_town03_SS.png')
    plt.show()
    all_faultidx = np.where(AccLabel!=0)[0]
    plt.figure(2)
    plt.plot(t[all_faultidx],AccLabel[all_faultidx],'ro', label='ground truth')   
    plt.plot(t[all_faultidx], acc_predlabel[all_faultidx],'b+', label='predicted faults')   
    plt.xlabel('Sample id')
    plt.ylabel('Class label')
    plt.legend()
    plt.yticks([0, 1, 2, 3, 4, 5])
    plt.savefig('results_town03_SS_miss.png')
    plt.show()
    plt.figure(3)
    cycol = cycle('bgrcmk')
    colors = ['b','g','r','c','m','y', 'k', 'gray','pink']
    fault_types = list(fea_relinfo.keys())
    figs, axes = plt.subplots(3,3,sharex=True,sharey=True)
    sensors = ['Compass','GPS','Imu']
    lines = []
    labels = []
    for i in range(3):
        for j in range(3):
            idx = i*3+j
            key = fault_types[idx]
            s_idx = np.argsort(fea_relinfo[key])[-1:]
            s_lw = np.asarray([1.5,1.5,1.5])
            s_ec = np.asarray(['white','white','white'])
            s_ec[s_idx] = 'black'
            s_lw[s_idx] = 3.0
            axes[i,j].barh(np.arange(len(sensors)),fea_relinfo[key],label=str(key),color=colors[idx],edgecolor=s_ec,linewidth=s_lw)
            line, label = axes[i,j].get_legend_handles_labels()
            lines += line
            labels += label
    plt.legend(lines, labels, loc = 'upper right',bbox_to_anchor=(1.8, 3.5))
    plt.setp(axes,yticks=[0, 1, 2],yticklabels = sensors)
#    plt.suptitle('Sensor faulty probabilities with respect a specific fault type')
#    plt.savefig('sensor_town03_SS.png')
    plt.show()

