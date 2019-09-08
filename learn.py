#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from models.GSEM import GSEM



## definition of paths

negative_multipliers = [1, 5, 10, 15, 20, 30, 50, 100]

dataset_base_path = './data/'
result_path = './results/'
project_name = 'msb201126'

dataset_prefix = dataset_base_path
disease_path = dataset_prefix + '{0}_disease_similarities.pickle'.format(project_name)

split_prefix = dataset_base_path + '{0}_splits/'.format(project_name)
split_paths = [split_prefix + 'ratio_{0}/splits.pickle'.format(negative_multiplier) for negative_multiplier in negative_multipliers]

indication_path = dataset_prefix + '{0}_indications.pickle'.format(project_name)




## utils

def load(path):
    '''
        Loads any pickle file from disk.
    '''
    with open(path, 'rb') as handle:
        loaded = pickle.load(handle)

    return loaded


def threshold_similarities(matrix, percentile=75, threshold=None, remove_selfloops=False):
    '''
        Thresholds links in similarity matrix.
        All those with a strength lower than the one computed
        for the passed percentile will be cleaved off.
        As an alternative you can directly pass a score threshold.
    '''
    if remove_selfloops:
        matrix = matrix.copy()
        matrix[np.diag_indices(len(matrix))] = 0

    data = matrix.ravel()
    if threshold is None:
        threshold = np.percentile(data, percentile)

    indexer = np.where(matrix < threshold)
    new_matrix = matrix.copy()
    new_matrix[indexer] = 0.0

    return new_matrix




## load data

disease_similarities = load(disease_path)
indications = load(indication_path)




## pick similarity

disease_target = 'phenotype'




## fitting routine

def cross_validate(splits, data, regularizers, fitting_params):

    num_folds = len(splits)
    I, F_col = data
    tol, b, maxiter, mask_learning_sets = fitting_params

    aupr_scores = list()
    for i in range(num_folds):

        indexers = splits[i]
        model = GSEM(
            I,
            indexers,
            regularizers,
            F_col,
            complete_output_smoothing=False)

        fitted_model = model.fit(tol, maxiter, mask_learning_sets, verbose=False)
        aupr_scores.append(model.aupr(mask_learning_sets))

        print('... split {0} completed. ({1:.2f} AUPR)'.format(i, aupr_scores[-1]['test']))

    return aupr_scores




## plotting routine

def plot_results(scores_dict, xlabel, ylabel, path):

    codes = sorted(scores_dict.keys())
    scores = [scores_dict[code] for code in codes]
    x = np.arange(len(scores))
    means = [score.mean() for score in scores]
    plt.figure(dpi=300)
    for c, code in enumerate(codes):
        score_array = scores[c]
        n = len(score_array)
        mean = score_array.mean()
        std = score_array.std()
        plt.errorbar([c], [mean], yerr=[std], ecolor='black', elinewidth=0.75)
        plt.scatter([c]*n, score_array, s=8, facecolors='none', edgecolors='black', linewidths=0.2)
        plt.fill_between([c], [score_array.min()], [score_array.max()], lw=0.3, color='grey')
    plt.plot(x, means, '-', linewidth=0.75, color='black')
    plt.xticks(x, codes, fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path, format='png')
    plt.close()




## fitting parameters

tol = 1e-3
b = 1e-2
maxiter = int(3e3)
mask_learning_sets = {'test'}

fitting_params = (tol, b, maxiter, mask_learning_sets)




## model parameters and data

regularizers = {
    'alpha': 1.0,
    'beta': 0.1,
    'lamda': 0.0,
    'gamma': 1e4}

tau = 0.25
disease_weight_matrix = threshold_similarities(disease_similarities[disease_target], threshold=tau, remove_selfloops=False)

data = (indications, disease_weight_matrix)




## learning

print('Performing 10-folds cross-validation across different settings...')
print('\n-----------------------------------------------------------------\n')

scores = dict()
for s, split_path in enumerate(split_paths):

    print('\n=======------- ratio {0}:1 -------=======\n\n'.format(negative_multipliers[s]))

    splits = load(split_path)
    aupr_scores = cross_validate(splits, data, regularizers, fitting_params)

    test_scores = np.asarray([score['test'] for score in aupr_scores])
    scores[negative_multipliers[s]] = test_scores
    print('\n\t mean TEST AUPR: {0}\n\n'.format(test_scores.mean()))




## save results

plot_results(scores, 'negative multiplier', 'AUPR scores', result_path+'plot.png')
with open(result_path+'scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle)

print('------------------------------------\n')
print('Results saved at folder {0} .'.format(result_path))
