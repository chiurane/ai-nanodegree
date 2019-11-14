import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Now BIC = -2 * logL + p * logN
        # Where L: Loglikelihood
        # p is the number of parameters
        # n is the of data points
        # TODO implement model selection based on BIC scores
        # raise NotImplementedError
        # bics = [(n, ((-2) * self.base_model(n).score(self.X, self.lengths)) + \
        #         ((n * n + 2 * n * len(self.X[0]) - 1) * np.log(self.X.shape[0]))) \
        #        for n in range(self.min_n_components, self.max_n_components + 1)]
        #best_bic = min(bics, key = lambda x: x(1)) # Get the n with the smallest bic.
        #return self.base_model(best_bic[0])
        max_bic = float('inf') # We are looking for the smallest BIC. Smallest BIC is a good thing.
        best_model = None
        # Go through components, recomputing the model until the smallest is found
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths) # Compute the log likelihood here
                p = (n * n + 2 * n * len(self.X[1]) - 1)
                bic = (-2 * logL) + (p * np.log(self.X.shape[0]))
                if bic < max_bic:
                    max_bic = bic
                    best_model = model
            except:
                continue
        if not best_model: # Just in case we didnt find anything, return default based on n_constants
            return self.base_model(self.n_constant)
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # raise NotImplementedError
        # The highest DIC is the best or desired one
        # So we want to iterate through all our models, then choose the highest DIC one.
        smallest_dic = float('-inf')
        best_model = None # This is the one I will return with the highest DIC
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths) # The log likelihood
                other_total_logL = 0
                for word in self.words:
                    if word != self.this_word:
                        x_other, lengths_other = self.hwords[word]
                        other_total_logL += model.score(x_other, lengths_other)
                logL_avg = other_total_logL/(len(self.words) - 1) # Average of all the loglikelihoods
                curr_dic = logL - logL_avg
                if curr_dic > smallest_dic:
                    smallest_dic = curr_dic
                    best_model = model
            except:
                continue
        if not best_model: # In case our model None
            best_model = self.base_model(self.n_constant)
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # raise NotImplementedError
        smallest_cv = float('-inf')
        best_model = None
        split_method = KFold()
        for n in range(self.min_n_components, self.max_n_components + 1):
            fold_scores = list() # We need to compute the average fold score
            try:
                for cv_train_idx, cv_test_idx, in split_method.split(self.sequences):
                    X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, length_train)
                    logL = model.score(X_test, length_test)
                    fold_scores.append(logL)
                avg_fold_score = np.mean(fold_scores)
                if avg_fold_score > smallest_cv:
                    smallest_cv = avg_fold_score
                    best_model = model
            except:
                pass
        if not best_model:
            best_model = self.base_model(self.n_constant)
        return best_model