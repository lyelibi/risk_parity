# risk_parity
Replication Study: Hierarchical Risk Parity, with synthetic correlation matrices, and alternatively on real data with Agglomerative Likelihood Clustering

''' Conclusions on Synthetic Data:
    The minimum variance portfolio method outperforms in sample but has the worst performance
    out of sample. a strong signal of an overfitted estimator.
    Naive Risk is theoretical the upper bound of hierarchical Risk Parity. In our trials, we see that
    if there is a block structure in the covariance matrices HRP dominates both the equal weight, and the Naive RP
    portfolio both in and out of sample in our experience.
    
    We note how that logically in a data set where there number of cluster must be relatively high
    for the hierarchy to significantly impact the out of sample volatility. Also the size of clusters
    must be large, small clusters tend to equalize HRP and NRP.
    '''
    
    ''' Conclusions on real financial market data:
    On this data we have we are able to show again that the minimum variance portfolio performs
    poorly out of sample. In this instance HRP with single linkage and HRP with ALC performs similarly to
    Naive HRP. Unlike in our synthetic trials. One possible explanation is non-stationarity: The out of sample returns
    in the previous section are generated from the same correlation struture. It is however not given that the correlation
    matrix is static through time. Another possible interpretation could be just that there are not enough clusters, not 
    enough hierarchy in this data-set for hierarchical risk parity to dominate Naive RP. Finally this data is one path, there is a need to generate more paths to       estimate the statistical advantage of portfolio allocation using HRP
    '''
