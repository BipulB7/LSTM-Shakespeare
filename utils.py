import numpy as np

def get_tensor(sentence, maximum_seq_length, char_indices):
    x = np.zeros((1, maximum_seq_length, len(char_indices)), dtype=bool)
    for t, ch in enumerate(sentence[-maximum_seq_length:]):
        x[0, t, char_indices[ch]] = 1
    return x

def get_tensor_emb(sentence, maximum_seq_length, voc):
    x = np.zeros((1, maximum_seq_length), dtype=int)
    for t, ch in enumerate(sentence[-maximum_seq_length:]):
        x[0, t] = voc.get(ch, 0)
    return x

def sample(predictions, temperature=1.0):
    preds = np.asarray(predictions).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    probs = exp_preds / np.sum(exp_preds)
    choice = np.random.multinomial(1, probs, 1)
    return np.argmax(choice)
