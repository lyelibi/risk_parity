''' Replication study hierarchical Risk Parity based on the work by Gautier Marti
at https://gmarti.gitlab.io/qfin/2018/10/02/hierarchical-risk-parity-part-1.html

and on the original paper by Marcos Lopez De Prado https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
'''


import numpy as np

def kron(a,b):
    if a==b:
        return 1
    else:
        return 0
    
def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

import matplotlib.pyplot as plt
from timeseries_generator.onefactor import onefactorwithvar
from alc.alc import alc

''' We want to investigate how Hierarchical Risk Parity performs compared to the minimum variance,
the equal weight, and the naive risk partity portfolios in and out of sample.'''

# np.random.seed(0)


''' By using a one-factor correlated cluster model we generate data-set of correlated time-series:
    every clusters has its own variance. Our expectations being that all stocks in the same cluster have
    the same variance'''

cluster_size, clusters_number,length, coupling_parameter=[10,100,2000,.99]
N = cluster_size * clusters_number
ret, alpha, stds = onefactorwithvar(cluster_size, clusters_number, length, coupling_parameter, var=None)

covar = np.cov(ret)

plt.figure()
plt.pcolormesh(covar)
plt.colorbar()
plt.title('Covariance matrix')
plt.show()
plt.savefig('true_covariance.png')


''' sample returns from the covariance matrices '''
alphas_returns = np.random.multivariate_normal(
    np.zeros(N), cov=covar, size=500).T

plt.figure(figsize=(20, 10))
for i in range(N):
    plt.plot(range(500),alphas_returns.cumsum(axis=1)[i])
plt.title('Performance of the multivariate returns', fontsize=24)
plt.show()
plt.savefig('cumul_multivariate.png')



''' estimate the correlations and covariance '''
estimate_correl = np.corrcoef(alphas_returns)
estimate_covar = np.cov(alphas_returns)

plt.figure()
plt.pcolormesh(estimate_correl)
plt.colorbar()
plt.title('Estimated correlation matrix Multivariate Normal')
plt.show()
plt.savefig('estim_cor_multivar.png')

plt.figure()
plt.pcolormesh(estimate_covar)
plt.colorbar()
plt.title('Estimated covariance matrix Multivariate Normal')
plt.show()
plt.savefig('estim_cov_multivar.png')



''' les choses serieuses '''
''' define the different weightings '''
import pandas as pd
def compute_HRP_weights(covariances, alpha):
    clusters = np.unique(alpha)

    weights = np.empty(len(alpha))
    com_weights = pd.DataFrame(1, index=clusters,columns=['0'])
    com_var = pd.DataFrame(1, index=clusters,columns=['0'])
    fcovar = pd.DataFrame(covariances)
    for i in clusters:
        community = duplicates(alpha,i)
        cluster_cov = fcovar[community].loc[community]
        inv_diag = 1 / np.diag(cluster_cov.values)
        parity_w = inv_diag * (1 / np.sum(inv_diag)) # Normalization
        weights[community] = parity_w
        cluster_var = np.dot(parity_w, np.dot(cluster_cov, parity_w))
        com_var.loc[i] = cluster_var
    
    ''' obtain the weight of individual clusters '''
    inv_com_var = 1 / com_var.values
    com_weights.loc[clusters] = inv_com_var / np.sum(inv_com_var)
    
    for i in clusters:
        community = duplicates(alpha,i)
        weights[community] = weights[community]*com_weights.loc[i]['0']
                
    return weights

'''  Define the minimum variance, risk parity, and equal weights functions '''

def compute_MV_weights(covariances):
    inv_covar = np.linalg.inv(covariances)
    u = np.ones(len(covariances))
    
    return np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))


def compute_RP_weights(covariances):
    weights = (1 / np.diag(covariances)) 
    
    return weights / sum(weights)


def compute_unif_weights(covariances):
    
    return [1 / len(covariances) for i in range(len(covariances))]

print('''IN SAMPLE: Estimate annualized portfolio volatilities''')
# in-sample HRP annualized volatility

# using estimated covariance

HRP_weights = compute_HRP_weights(estimate_covar, alpha)
HRP_weights_returns = np.array([HRP_weights[i] * alphas_returns[i] for i in range(N)])
print('HRP', round(HRP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# in-sample 1 / N annualized volatility
unif_weights = compute_unif_weights(estimate_covar)
unif_weights_returns = np.array([unif_weights[i] * alphas_returns[i] for i in range(N)])

print('Equal Weight',round(unif_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# in-sample naive risk parity normalized(1 / vol) volatility

# using estimated covariance matrix
RP_weights = compute_RP_weights(estimate_covar)
RP_weights_returns = np.array([RP_weights[i] * alphas_returns[i] for i in range(N)])

print('Naive Risk Parity' , round(RP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# in-sample Minimum Variance annualized volatility

# using estimated covariance matrix
MV_weights = compute_MV_weights(estimate_covar)
MV_weights_returns = np.array([MV_weights[i] * alphas_returns[i] for i in range(N)])

print('Minimum Variance',round(MV_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))


print('''OUT OF SAMPLE: : Estimate annualized portfolio volatilities''')
nb_observations = 252

out_of_sample_alphas = np.random.multivariate_normal(
    np.zeros(N), cov=covar, size=nb_observations).T

# in-sample HRP annualized volatility

# using estimated covariance

HRP_weights = compute_HRP_weights(estimate_covar, alpha)
HRP_weights_returns = np.array([HRP_weights[i] * out_of_sample_alphas[i] for i in range(N)])
print('HRP',round(HRP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# in-sample 1 / N annualized volatility
unif_weights = compute_unif_weights(estimate_covar)
unif_weights_returns = np.array([unif_weights[i] * out_of_sample_alphas[i] for i in range(N)])

print('Equal Weight', round(unif_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# in-sample naive risk parity normalized(1 / vol) volatility using estimated covariance matrix
RP_weights = compute_RP_weights(estimate_covar)
RP_weights_returns = np.array([RP_weights[i] * out_of_sample_alphas[i] for i in range(N)])

print('Naive Risk Parity', round(RP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# in-sample Minimum Variance annualized volatility using estimated covariance matrix
MV_weights = compute_MV_weights(estimate_covar)
MV_weights_returns = np.array([MV_weights[i] * out_of_sample_alphas[i] for i in range(N)])

print('Minimum Variance', round(MV_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
''' Conclusions:
    The minimum variance portfolio method outperforms in sample but has the worst performance
    out of sample. a strong signal of an overfitted estimator.
    Naive Risk is theoretical the upper bound of hierarchical Risk Parity. In our trials, we see that
    if there is a block structure in the covariance matrices HRP dominates both the equal weight, and the Naive RP
    portfolio both in and out of sample in our experience.
    
    We note how that logically in a data set where there number of cluster must be relatively high
    for the hierarchy to significantly impact the out of sample volatility. Also the size of clusters
    must be large, small clusters tend to equalize HRP and NRP.
    '''

'''  The next phase consists in applying the same routine on real stock market
data. We have obtained 5 years of SP500 returns (447 stocks). We split the data
into a train and test set such that the covariances are estimated on the train set,
the portfolio weights are estimated in sample, then use on out of sample returns'''


''' Gautier Marti in his original study used the hierarchical clustering to obtain
the hierarchy in the data. Here we reproduce it but further down we would like to compare
the result of to another hierarchical clustering algorithm called Agglomerative Likelihood Clustering
https://arxiv.org/abs/1908.00951

'''
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def seriation(Z, N, cur_index):
    """Returns the order implied by a hierarchical tree (dendrogram).
    
       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.
       
       :return: The order implied by the hierarchical tree Z.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))

    
def compute_serial_matrix(dist_mat, method="ward"):
    """Returns a sorted distance matrix.
    
       :param dist_mat: A distance matrix.
       :param method: A string in ["ward", "single", "average", "complete"].
        
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a,b = np.triu_indices(N, k=1)
    seriated_dist[a,b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

def compute_HRP_marti(covariances, res_order):
    weights = pd.Series(1, index=res_order)
    clustered_alphas = [res_order]

    while len(clustered_alphas) > 0:
        clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        for subcluster in range(0, len(clustered_alphas), 2):
            left_cluster = clustered_alphas[subcluster]
            right_cluster = clustered_alphas[subcluster + 1]

            left_subcovar = covariances[left_cluster].loc[left_cluster]
            inv_diag = 1 / np.diag(left_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

            right_subcovar = covariances[right_cluster].loc[right_cluster]
            inv_diag = 1 / np.diag(right_subcovar.values)
            parity_w = inv_diag * (1 / np.sum(inv_diag))
            right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

            alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

            weights[left_cluster] *= alloc_factor
            weights[right_cluster] *= 1 - alloc_factor
            
    return weights
''''''''''''''''''''''''''''''''''''''''''''''''




print('Real Market Data''')
data = pd.read_csv('sp500_1250d.csv', index_col ='Name')
name = np.array((data.index))
# ''' Industry and Sector Coloring'''
# keys = pd.read_csv('key_stock.csv', index_col ='Name')
# key = keys['Sector']
data = np.diff(np.log(data))
N = data.shape[0]
k = 500
print('start of the time-series',k)
train_data = data[:,:k]
test_data = data[:,k:k+252]

'''  estimate covariance and correlation matrices '''
train_covar = np.cov(train_data)
train_cor = np.corrcoef(train_data)


''' Use the single linkage Hierarchcial Clustering algorithm to get the hierarchy for HRP'''

distances = np.around(np.sqrt((1 - train_cor) / 2),2)
np.fill_diagonal(distances,0)
ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(distances, method='single')

''' Here we use ALC to get the hierarchy in the data'''
alphar = alc(train_cor)




print('Market Data: IN SAMPLE')

HRP_marti_weights = compute_HRP_marti(pd.DataFrame(train_covar), res_order)
HRP_marti_returns = np.array([HRP_marti_weights[i] * train_data[i] for i in range(N)])

print('HRP Single Linkage', round(HRP_marti_returns.sum(axis=0).std() * np.sqrt(252),
            2))

HRP_weights = compute_HRP_weights(train_covar, alphar)
HRP_weights_returns = np.array([HRP_weights[i] * train_data[i] for i in range(N)])
print('HRP ALC',round(HRP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# in-sample 1 / N annualized volatility
unif_weights = compute_unif_weights(train_covar)
unif_weights_returns = np.array([unif_weights[i] * train_data[i] for i in range(N)])

print('Equal Weight',round(unif_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# in-sample naive risk parity normalized(1 / vol) volatility
# using estimated covariance matrix
RP_weights = compute_RP_weights(train_covar)
RP_weights_returns = np.array([RP_weights[i] * train_data[i] for i in range(N)])

print('Naive RP', round(RP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# in-sample Minimum Variance annualized volatility
# using estimated covariance matrix
MV_weights = compute_MV_weights(train_covar)
MV_weights_returns = np.array([MV_weights[i] * train_data[i] for i in range(N)])

print('Min Var', round(MV_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))





print('Market Data : OUT OF SAMPLE')

HRP_marti_weights = compute_HRP_marti(pd.DataFrame(train_covar), res_order)
HRP_weights_returns = np.array([HRP_marti_weights[i] * test_data[i] for i in range(N)])
print('HRP with Linkage',round(HRP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

HRP_weights = compute_HRP_weights(train_covar, alphar)
HRP_weights_returns = np.array([HRP_weights[i] * test_data[i] for i in range(N)])
print('HRP with ALC',round(HRP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# 1 / N annualized volatility
unif_weights = compute_unif_weights(train_covar)
unif_weights_returns = np.array([unif_weights[i] * test_data[i] for i in range(N)])

print('Equal Weight',round(unif_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
# naive risk parity normalized(1 / vol) volatility
# using estimated covariance matrix
RP_weights = compute_RP_weights(train_covar)
RP_weights_returns = np.array([RP_weights[i] * test_data[i] for i in range(N)])

print('Naive RP', round(RP_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))

# in-sample Minimum Variance annualized volatility
# using estimated covariance matrix
# MV_weights = compute_MV_weights(train_covar)
MV_weights_returns = np.array([MV_weights[i] * test_data[i] for i in range(N)])

print('Min Var', round(MV_weights_returns.sum(axis=0).std() * np.sqrt(252),
            2))
    
''' Conclusions:
    On this data we have we are able to show again that the minimum variance portfolio performs
    poorly out of sample. In this instance HRP with single linkage and HRP with ALC performs similarly to
    Naive HRP. Unlike in our synthetic trials. One possible explanation is non-stationarity: The out of sample returns
    in the previous section are generated from the same correlation struture. It is however not given that the correlation
    matrix is static through time. Another possible interpretation could be just that there are not enough clusters, not 
    enough hierarchy in this data-set for hierarchical risk parity to dominate Naive RP.
    '''
    