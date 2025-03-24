import numpy as np

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
          chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)
    def encode(self, C, nb_rows):
        """One-hot encode given string C.
        # Arguments
          C: string, to be encoded.
          nb_rows: Number of rows in the returned one-hot encoding. This is
          used to keep the # of rows for each data the same via padding.
        """
        x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(C):
            # Check if the character is in the char2index dictionary
            if c in self.char2index:
                x[i, self.char2index[c]] = 1.0
            else:
                # Handle unseen characters (you can choose to ignore or handle differently)
                print(f"Unseen character: {c}")
        return x

    # def encode(self, C, nb_rows):
    #     """One-hot encode given string C.
    #     # Arguments
    #       C: string, to be encoded.
    #       nb_rows: Number of rows in the returned one-hot encoding. This is
    #       used to keep the # of rows for each data the same via padding.
    #     """
    #     x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
    #     for i, c in enumerate(C):
    #         x[i, self.char2index[c]] = 1.0
    #     return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
          x: A vector or 2D array of probabilities or one-hot encodings,
          or a vector of character indices (used with `calc_argmax=False`).
          calc_argmax: Whether to find the character index with maximum
          probability, defaults to `True`.
        """
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars

    def sample_multinomial(self, preds, temperature=1.0):
        """Sample index and character output from `preds`,
        an array of softmax probabilities with shape (1, 1, nb_chars).
        """
        # Reshaped to 1D array of shape (nb_chars,).
        preds = np.reshape(preds, len(self.chars)).astype(np.float64)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        index = np.argmax(probs)
        char  = self.index2char[index]
        return index, char