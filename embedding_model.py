import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation
from keras.optimizers import Adam
from callbacks.text_generation_callbacks import end_epoch_generate, early_stopping, reduce_lr
from src.utils import get_tensor_emb, sample



#  X_emb, y_emb as per Q10
X_emb = np.zeros((len(sentences), maximum_seq_length), dtype=int)
y_emb = np.zeros(len(sentences), dtype=int)
for i, sent in enumerate(sentences):
    for t, ch in enumerate(sent):
        X_emb[i, t] = char_indices[ch]
    y_emb[i] = char_indices[next_char[i]]

# Model building
model_emb_m2m = Sequential()
model_emb_m2m.add(Embedding(input_dim=len(chars), output_dim=32, input_length=maximum_seq_length))
model_emb_m2m.add(LSTM(128))
model_emb_m2m.add(Dense(len(chars), activation='softmax'))
model_emb_m2m.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001))

#Traininig
model_emb_m2m.fit(X_emb, y_emb,
                  batch_size=64,
                  epochs=5,
                  validation_split=0.2,
                  callbacks=[end_epoch_generate, early_stopping, reduce_lr])

# gen
def generate_next_emb(model, text, num_generated=120, temperature=1.0):
    generated = text
    sentence = text[-maximum_seq_length:]
    for _ in range(num_generated):
        x = get_tensor_emb(sentence, maximum_seq_length, char_indices)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated
