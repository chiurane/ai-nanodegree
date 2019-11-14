import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    # raise NotImplementedError
    # We need the XLengths and sequences from the test_set first
    Xlengths = test_set.get_all_Xlengths()
    sequences = test_set.get_all_sequences()

    # Now we iterate and build our probabilities and and guesses lists
    for sequence in sequences:
        best_guess = None
        best_logL = float('-inf') # Start with the smallest possible logL, we want to increment this to determine best guess
        prob_dict = dict() # Dictionary of probabilities, to be added to probablilities
        X, xlengths = Xlengths[sequence]
        # Now we go through all words and their models
        for guess, model in models.items():
            try:
                logL = model.score(X, xlengths)
            except:
                logL = float('-inf') # Set it to minus infinity if an error occurs
            prob_dict[guess] = logL
            if logL > best_logL:
                best_logL = logL
                best_guess = guess
        # Now append the processed dictionary
        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return probabilities, guesses


