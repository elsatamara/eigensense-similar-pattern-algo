import numbers
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
import pandas as pd
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import statistics
import datetime
import functools
import multiprocess
import time, datetime

class ReversePriorityQueue(PriorityQueue):

    def put(self, tup):
        newtup = tup[0] * -1, tup[1]
        PriorityQueue.put(self, newtup)

    def get(self):
        tup = PriorityQueue.get(self)
        newtup = tup[0] * -1, tup[1]
        return newtup

def tsCandidate(data, slidingWindow, interval, D_norm):
   
    len_Data = len(data)

    for i in range(0, len_Data-slidingWindow, interval):
        DD = data[i:i + slidingWindow]       
        DD_mean = statistics.mean(DD)
        DD_std = statistics.stdev(DD)
        if DD_std == 0:
            DD_norm = (DD - DD_mean)
        else:
            DD_norm = (DD - DD_mean)/DD_std
        D_norm.append([i,DD_norm])
        
    return D_norm

def DTWComputation(ts_query, ts_candidate, d_cb, Res, k, idx):

    ''' Return the updated priority queue that contains the most similar subsequences

        Parameters
        ----------
        ts_query : array_like
                Sample sequence
        ts_candidate : array_like
                    Subsequence                
        d_cb : int
            Tolerace for the distance 
        Res : priority queue 
            prioirty queue with k entires 
            each entry = <distance, sequence>
        k : int
            Number of data sequence that are most similar to ts_query
        idx : int
            Starting index of the subsequence in the database

        

        Returns
        -------
        Res : priority queue 
            prioirty queue with k entires 
            each entry = <distance, sequence>
    '''
        
    
    d_lbs_exact, path = fastdtw(ts_query, ts_candidate)
    
    d_lbs_exact = d_lbs_exact*len(ts_query)/len(ts_candidate)
 
    if Res.qsize() < k:
        Res.put((d_lbs_exact, idx))

    if Res.qsize() == k:
             
        if d_lbs_exact > d_cb: return Res
        else:
            Res.get()
        Res.put((d_lbs_exact, idx))

    return Res

def LB_Yi(ts_candidate, ts_query, d_cb):
    ''' return the LB_Yi distance between the sample and subsequence

        Parameters
        ----------
        ts_candidate : array_like
                    Subsequence
        ts_query : array_like
                Sample sequence
        d_cb : int
            Tolerace for the distance 

        Returns
        -------
        d_cb : int
            LB_Yi distance between the sample and subsequence
        tmp : bool
            tmp == False means early abondon occurs as the lower bounding distance 
                exceed d_cb
    '''
    
    
    d_lb = 0
    tmp = True
    min_q, max_q = min(ts_query), max(ts_query)
    min_c, max_c = min(ts_candidate), max(ts_candidate)
    flag = True if max_q > max_c else False
    x = ts_query if flag == True else ts_candidate
    y = ts_candidate if flag == True else ts_query
    
    min_x, max_x = min(x), max(y)
    min_y, max_y = min(y), max(y)
    
    if min_x > max_y:
        d1, d2  = 0, 0
        for xi in x:
            d1 += abs(xi - max_y) 
            if d1 > d_cb: 
                d1 = -float('inf')
                break
        for yi in y:
            d2 += abs(yi - min_x) 
            if d2 > d_cb: 
                d2 = -float('inf')
                break
        if d1 == -float('inf') and d2 == -float('inf'):
            tmp = False
        else:
            d_lb = max(d1, d2)
    elif min_x < min_y:
        for xi in x:
            if xi > max_y:
                d_lb += abs(xi - max_y)
            elif xi < min_y:
                d_lb += abs(xi - min_y)
            if d_lb > d_cb:
                tmp = False
                break
    elif min_x <= max_y and min_x >= min_y:
        for xi in x:
            if xi > max_y:
                d_lb += abs(xi - max_y)
            if d_lb > d_cb:
                tmp = False
                break
        for yi in y:
            if yi < min_x:
                d_lb += abs(yi - min_x)
            if d_lb > d_cb:
                tmp = False
                break
    
    return d_lb, tmp
                
def kNNsearch_LB_Yi(ts_query, D, k, numData, method = 'EA', d_cb = float('inf')):
    ''' Return the k sequences that are most similar to 
                    the sample sequence measured by DTW distance

        Parameters
        ----------
        ts_query : array_like
            Sample sequence data points
        D : DataFrame (column = [Starting Idx, Subsequence])
            DataFrame contains a list of subsequences
            with two columns 'Starting Idx' and 'Subsequence', 'Starting Idx' stores the starting
            index of the subsequence in the dataset, 'Subsequence' is the corresponding subsequences
        k : int
            Number of data sequence that are most similar to ts_query
        method: string
            method takes two possible inputs, 'EA' or 'Complete'
            'EA': using early abondon method and get top k most similar subsequences
            'Complete': compute the DTW for all subsequences and output top k most similar subsequences
        d_cb: int
            Tolerace for the distance 

        Returns
        -------
        res : array_like 
            array with k entires 
            each entry = [distance score, [[timestamp0, datapoint0], [timestamp1, datapoint1]...]
    '''
    


    Res = ReversePriorityQueue()

    
    #z-normaliztion for the sample sequence

    ts_query_mean = statistics.mean(ts_query)
    ts_query_std = statistics.stdev(ts_query)
    ts_query_norm = [(i - ts_query_mean)/ts_query_std for i in ts_query]
    
    data = D[0]
    startIndex = D[1]

    if method == 'Complete':
        
        
        
        for idx in range(len(data)):
            
            ts = data['Subsequence'].iloc[idx]
            ts = ts.replace('[', '').replace(']', '')
            ts = np.array(ts.split())
            ts_candidate = [float(i) for i in ts]
            
            d_lbs_exact, path = fastdtw(ts_query_norm, ts_candidate)
            d_lbs_exact = d_lbs_exact*len(ts_query_norm)/len(ts_candidate)
            Res.put((d_lbs_exact, idx + startIndex*numData))
            
            
            
    
    if method == 'EA':

        for idx in range(len(data)):
            ts = data['Subsequence'].iloc[idx]
            ts = ts.replace('[', '').replace(']', '')
            ts = np.array(ts.split())
            ts_candidate = [float(i) for i in ts]
        
                
            d_lb, tmp = LB_Yi(ts_candidate, ts_query_norm, d_cb)
                
            if d_lb <= d_cb and tmp == True:
                Res = DTWComputation(ts_query_norm, ts_candidate, d_cb, Res, k, idx + startIndex*numData)

            d_cb = -Res.queue[0][0] if not Res.empty() else d_cb
    
    
    return Res

def TopK(allSubsequence, D, k):

    res = []
    for i in range(k):
        
        x = allSubsequence[i]
        dis_score = x[0]
        idx = x[1]
        data = []
        
        ts = D['Subsequence'].iloc[idx]
        ts = ts.replace('[', '').replace(']', '')
        ts = np.array(ts.split())
        ts_candidate = [float(i) for i in ts]
        
        t_tmp = D['Time'].iloc[idx]
        t_tmp = t_tmp.replace('[', '').replace(']', '')
        t_tmp = np.array(t_tmp.split())
        timestamp = []
        for j in range(0,len(t_tmp),2):
            strtotime = t_tmp[j]+ ' ' + t_tmp[j+1]
            strtotime = strtotime[1:len(strtotime)-1]
            strtotime = datetime.datetime.strptime(strtotime, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            timestamp.append(strtotime)
        
        
        for k in range(len(ts_candidate)):
            data.append([timestamp[k], ts_candidate[k]])
    
        res.append([dis_score, data])
    return res

def parallelComputing(ts_query, D, k, method = 'EA', d_cb = float('inf')):
    
    ''' Return the k sequences that are most similar to 
                    the sample sequence measured by DTW distance, all sequences if 'Complete' method is used
                    and searching time 

        Parameters
        ----------
        ts_query : array_like
            Sample sequence data points
        D : DataFrame (column = [Starting Idx, Subsequence])
            DataFrame contains a list of subsequences
            with two columns 'Starting Idx' and 'Subsequence', 'Starting Idx' stores the starting
            index of the subsequence in the dataset, 'Subsequence' is the corresponding subsequences
        k : int
            Number of data sequence that are most similar to ts_query
        method: string
            method takes two possible inputs, 'EA' or 'Complete'
            'EA': using early abondon method and get top k most similar subsequences
            'Complete': compute the DTW for all subsequences and output top k most similar subsequences
        d_cb: int
            Tolerace for the distance 

        Returns
        -------
        topKSubsequence : array_like 
            array with k entires 
            each entry = [distance score, [[timestamp0, datapoint0], [timestamp1, datapoint1]...]
        allSubsequence : array_like
            array that stores the distance scores of all sequences if 'Complete' method is used,
            if 'EA' method is used, return the top k sequences outputted each processor
            each entry = [distance score, index of sequence in D]
        searchTime: float
            Time to complete all the distance computation/comparison
            
    '''
    
    
    allSubsequence = []
    topKSubsequence = []
    
    numData = int(len(D)/10)
    p = int(len(D)/numData)
    dataSplit = []
    for i in range(p):
        if i != p-1:
            dataSplit.append([D[i*numData:(i+1)*numData], i])
        else:
            dataSplit.append([D[i*numData:], i])
    
    start = time.time()
    partial = functools.partial(kNNsearch_LB_Yi, ts_query, k = k, numData = numData, method = method, d_cb = float('inf'))
    with multiprocess.Pool(processes=10) as pool:
        res = pool.map(partial, dataSplit)
    end = time.time()
    searchTime = end - start
    
    for i in range(len(res)):
        tmp = res[i]
        while not tmp.empty():
            x = tmp.get()
            allSubsequence.append([x[0], x[1]])
    allSubsequence = sorted(allSubsequence,key = lambda l:l[1], reverse=True) 
    topKSubsequence = TopK(allSubsequence, D, k)
    
    
    
    
    
    
    return topKSubsequence, allSubsequence, searchTime