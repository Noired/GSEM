from .base_model import *

class GSEM(Model):

    def __init__(self,
        indications,
        indexers,
        regularizers,
        column_affinity,
        complete_output_smoothing=True):

        super().__init__(indications, indexers, regularizers)
        self.F_col = column_affinity
        self.complete_output_smoothing = complete_output_smoothing

        return


    def _initialize_weights(self, weight_variance=1e-2):
        '''
            Initializes learning weights and returns
            a dictionary containing them, to be used in
            model fitting.
        '''
        weights = np.random.uniform(0, np.sqrt(weight_variance), (self.m, self.m))
        weight_dict = {'W': weights}

        return weight_dict


    def _prepare_data(self, mask_learning_sets):
        '''
            Masks model data and returns them in the
            form of a data dictionary. This is supposed
            to be called prior to learning.
        '''
        inputs = self.inputs
        masked_inputs = self._mask(mask_learning_sets, inputs)
        input_covariance = np.dot(masked_inputs.T, masked_inputs)
        column_affinity_degree_matrix = np.diag(self.F_col.sum(1))
        self.Y = masked_inputs
        self.Y_cov = input_covariance
        self.D_col = column_affinity_degree_matrix
        self.I = np.eye(self.m)
        self.dir_reg = self.regularizers['alpha']
        self.l2_reg = self.regularizers['beta']
        self.l1_reg = self.regularizers['lamda']
        self.gamma = self.regularizers['gamma']

        return


    def _numerator(self, weight_dict, variable=None):
        '''
            Returns the multiplicative rule numerator.
        '''
        Y = self.Y
        Y_cov = self.Y_cov
        F_col = self.F_col
        W = weight_dict['W']
        dir_reg = self.dir_reg

        numerator = 0.0
        numerator += Y_cov
        if self.complete_output_smoothing:
            numerator += dir_reg * np.dot(Y_cov, np.dot(W, F_col))
        else:
            numerator += dir_reg * np.dot(W, F_col)
        return numerator


    def _denominator(self, weight_dict, variable=None):
        '''
            Returns the multiplicative rule denominator.
        '''
        Y = self.Y
        Y_cov = self.Y_cov
        F_col = self.F_col
        D_col = self.D_col
        I = self.I
        W = weight_dict['W']
        dir_reg = self.dir_reg
        l2_reg = self.l2_reg
        l1_reg = self.l1_reg
        gamma = self.gamma

        denominator = 0.0
        denominator += np.dot(Y_cov, W)
        denominator += l2_reg * W
        denominator += l1_reg
        denominator += gamma * I
        denominator += self.eps
        if self.complete_output_smoothing:
            denominator += dir_reg * np.dot(Y_cov, np.dot(W, D_col))
        else:
            denominator += dir_reg * np.dot(W, D_col)
        return denominator


    def _multiplicative_rule(self, weight_dict):
        '''
            Applies the model characteristic update rule.
            Returns the updated weights in the form of a
            dictionary and an aggregated measure of update
            delta.
        '''
        prev_weights = weight_dict['W']
        numerator = self._numerator(weight_dict)
        denominator = self._denominator(weight_dict)

        update_ratio = np.divide(numerator, denominator)
        new_weights = np.multiply(prev_weights, update_ratio)
        new_weights.clip(min = 0)
        weight_dict['W'] = new_weights
        dw = np.amax(np.abs(new_weights - prev_weights)) / \
            (self.sqrteps + np.amax(np.abs(prev_weights)))

        return weight_dict, dw


    def fit(self,
        tol=1e-2,
        max_iter=int(1e3),
        mask_learning_sets={'test'},
        verbose=False,
        monitor_every=False):

        return super().fit(tol, max_iter, mask_learning_sets, verbose, monitor_every)


    def losses(self, mask_learning_sets):
        '''
            Computes current model losses
            and returns them in the form of a dictionary.
        '''
        # retrieve targets, predictions, weights...
        T = self._mask(mask_learning_sets, self.targets)
        Y_hat = self.predict(mask_learning_sets)
        W = self.weight_dict['W']
        L = self.D_col - self.F_col

        # compute frobenius norm on fitting
        fitting_loss = np.linalg.norm(T - Y_hat, 'fro')**2
        diag_loss = self.gamma * np.trace(W)
        l1_loss = self.l1_reg * np.linalg.norm(W.ravel(), 1)
        l2_loss = 0.5 * self.l2_reg * np.linalg.norm(W, 'fro')**2
        if self.complete_output_smoothing:
            dir_loss = 0.5 * self.dir_reg * np.trace((Y_hat).dot(L).dot(Y_hat.T))
        else:
            dir_loss = 0.5 * self.dir_reg * np.trace((W).dot(L).dot(W.T))

        # pack'em up
        losses = {
            'fitting': fitting_loss,
            'diag': diag_loss,
            'l1': l1_loss,
            'l2': l2_loss,
            'dirichlet': dir_loss,
            'complete': fitting_loss + diag_loss + l1_loss + l2_loss + dir_loss}

        return losses


    def metrics(self, mask_learning_sets):
        '''
            Computes current model metrics
            and returns them in the form of a dictionary.
        '''
        aurocs = self.auroc(mask_learning_sets)
        auprs = self.aupr(mask_learning_sets)
        metrics = dict()
        for learning_set in mask_learning_sets:
            metrics[learning_set+'_auroc'] = aurocs[learning_set]
            metrics[learning_set+'_aupr'] = auprs[learning_set]

        return metrics


    def predict(self, mask_learning_sets={'test'}):
        '''
            Computes model's outputs.
        '''
        try:
            if mask_learning_sets is None:
                Y = self.inputs
            else:
                Y = self._mask(mask_learning_sets, self.inputs)
            W = self.weight_dict['W']
            predictions = np.dot(Y, W)

        except AttributeError:
            print("The model has not been fitted yet, sorry. Try to run the 'fit' method first.")
            return

        return predictions
