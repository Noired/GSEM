import numpy as np
from sklearn import metrics as met
from matplotlib import pyplot as plt

class Model(object):

    def __init__(self,
        indications,
        indexers,
        regularizers):

        self.indications = indications
        self.n, self.m = indications.shape

        self.regularizers = regularizers
        self.eps, self.sqrteps = self._epsilon()

        self.inputs = self.indications.copy()
        self.targets = self.indications.copy()

        self.indexers = indexers

        return


    def _get_indexer(self, target_learning_sets):
        '''
            Returns a where-kind indexer to a particular
            learning set.
        '''
        if type(target_learning_sets) == set:
            held_outs = [self.indexers[target] for target in target_learning_sets]
            rows = [held_out[0] for held_out in held_outs]
            cols = [held_out[1] for held_out in held_outs]
            indexer = (np.concatenate(rows), np.concatenate(cols))

        else:
            indexer = self.indexers[target_learning_sets]

        return indexer


    def _mask(self, mask_learning_sets, object_to_mask):
        '''
            Masks out the specified learning sets from
            object.
        '''
        if mask_learning_sets is None:
            result = object_to_mask
        else:
            mask = np.ones((self.n, self.m))
            for mask_set in mask_learning_sets:
                mask[self.indexers[mask_set]] = 0.0
            result = object_to_mask * mask

        return result


    def _initialize_weights(self, weight_variance=1e-2):
        '''
            Initializes learning weights and returns
            a dictionary containing them, to be used in
            model fitting.
        '''
        raise NotImplementedError


    def _epsilon(self):
        '''
            Returns machine precision (its sqrt).
        '''
        epsilon = np.finfo(float).eps
        sqrteps = np.sqrt(epsilon)
        return epsilon, sqrteps


    def _numerator(self, weight_dict, variable=None):
        '''
            Returns the multiplicative rule numerator.
        '''
        raise NotImplementedError


    def _denominator(self, weight_dict, variable=None):
        '''
            Returns the multiplicative rule denominator.
        '''
        raise NotImplementedError


    def _multiplicative_rule(self, weight_dict):
        '''
            Applies the model characteristic update rule.
            Returns the updated weights in the form of a
            dictionary and an aggregated measure of update
            delta.
        '''
        raise NotImplementedError


    def _prepare_data(self, mask_learning_sets):
        '''
            Masks model data and returns them in the
            form of a data dictionary. This is supposed
            to be called prior to learning.
        '''
        raise NotImplementedError


    def fit(self,
        tol=1e-2,
        max_iter=int(1e3),
        mask_learning_sets={'test'},
        verbose=False,
        monitor_every=None):
        '''
            Fits model to data.
        '''
        self._prepare_data(mask_learning_sets)
        weight_dict = self._initialize_weights()
        self.weight_dict = weight_dict

        if verbose:
            print('Learning starting ...')
        if monitor_every is not None:
            dw_trace = list()
            loss_traces = dict()
            metric_traces = dict()
        for iter in range(max_iter):

            # update completion status
            if verbose:
                print('\rIteration: {}    '.format(iter), end='')

            # update weights
            weight_dict, dw = self._multiplicative_rule(weight_dict)
            self.weight_dict = weight_dict

            # run monitoring
            if monitor_every is not None and monitor_every != 0:
                if np.mod(iter, monitor_every) == 0:

                    # update weight delta collection
                    dw_trace.append(dw)

                    # udpate losses collection
                    losses = self.losses(mask_learning_sets)
                    for key in losses:
                        if key not in loss_traces:
                            loss_traces[key] = list()
                        loss_traces[key].append(losses[key])

                    # udpate metrics collection
                    metrics = self.metrics(mask_learning_sets)
                    for key in metrics:
                        if key not in metric_traces:
                            metric_traces[key] = list()
                        metric_traces[key].append(metrics[key])

            if dw <= tol:
                if verbose:
                    print('\r... done (delta is {0:.4f} @iteration {1}).'.format(dw, iter))
                break

        self.weight_dict = weight_dict
        if monitor_every is not None and monitor_every != 0:
            self.dw_trace = dw_trace
            self.loss_traces = loss_traces
            self.metric_traces = metric_traces

        return self


    def losses(self, mask_learning_sets):
        '''
            Computes current model losses
            and returns them in the form of a dictionary.
        '''
        raise NotImplementedError


    def metrics(self, mask_learning_sets):
        '''
            Computes current model metrics
            and returns them in the form of a dictionary.
        '''
        raise NotImplementedError


    def predict(self, mask_learning_sets={'test'}):
        '''
            Computes model's outputs.
        '''
        raise NotImplementedError


    def _extract_targets_and_scores(self, target_learning_sets):
        '''
            Extracts targets and scores to be used to
            compute performance metrics.
        '''
        predictions = self.predict(mask_learning_sets=target_learning_sets)

        if target_learning_sets is None:
            targets = self.targets.ravel()
            scores = predictions.ravel()
            target_dict['(all)'] = targets
            score_dict['(all)'] = scores

        else:
            target_dict = dict()
            score_dict = dict()
            for target_set in target_learning_sets:
                indexer = self._get_indexer(target_set)
                target_dict[target_set] = self.targets[indexer]
                score_dict[target_set] = predictions[indexer]

        return target_dict, score_dict


    def auroc(self, target_learning_sets={'test'}):

        target_dict, score_dict = self._extract_targets_and_scores(target_learning_sets)
        metric_dict = {target_set: met.roc_auc_score(target_dict[target_set], score_dict[target_set]) for target_set in target_learning_sets}

        return metric_dict


    def aupr(self, target_learning_sets={'test'}):

        target_dict, score_dict = self._extract_targets_and_scores(target_learning_sets)
        metric_dict = {target_set: met.average_precision_score(target_dict[target_set], score_dict[target_set]) for target_set in target_learning_sets}

        return metric_dict
