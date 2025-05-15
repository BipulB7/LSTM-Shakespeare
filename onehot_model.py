import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam
from callbacks.text_generation_callbacks import end_epoch_generate, early_stopping, reduce_lr
from src.utils import get_tensor, sample

# assume sentences, next_char, char_indices, indices_char, maximum_seq_length, chars are already defined

# 1) Build model
model = Sequential()
model.add(LSTM(128,
               input_shape=(maximum_seq_length, len(chars)),
               return_sequences=True,
               dropout=0.2,
               recurrent_dropout=0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001))

# 2) Train
model.fit(X, y,
          batch_size=128,
          epochs=20,
          validation_split=0.2,
          callbacks=[end_epoch_generate, early_stopping, reduce_lr])

# 3) Generation functions
def generate_next(model, text, num_generated=120, temperature=1.0):
    generated = text
    sentence = text[-maximum_seq_length:]
    for _ in range(num_generated):
        x = get_tensor(sentence, maximum_seq_length, char_indices)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def generate_beam(model, text, beam_size=5, num_generated=120):
    sentence = text[-maximum_seq_length:]
    current_beam = [(0.0, [], sentence)]
    for _ in range(num_generated):
        all_beams = []
        for log_prob, seq, sent in current_beam:
            x = get_tensor(sent, maximum_seq_length, char_indices)
            preds = model.predict(x, verbose=0)[0]
            preds = np.maximum(preds, 1e-10)
            for idx in seq[-3:]:
                preds[idx] *= 0.5
            top_idx = np.argsort(preds)[-beam_size:]
            for idx in top_idx:
                np_log = log_prob + np.log(preds[idx])
                new_seq = seq + [idx]
                new_sent = sent[1:] + indices_char[idx]
                all_beams.append((np_log, new_seq, new_sent))
        all_beams.sort(key=lambda x: x[0], reverse=True)
        current_beam = all_beams[:beam_size]
    best = max(current_beam, key=lambda x: x[0])[1]
    return text + ''.join(indices_char[i] for i in best)
