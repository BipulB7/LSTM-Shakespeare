import numpy as np
from keras.callbacks import LambdaCallback, EarlyStopping, ReduceLROnPlateau
from src.utils import get_tensor, get_tensor_emb, sample
from src.onehot_model import generate_next
from src.embedding_model import generate_next_emb

end_epoch_generate = LambdaCallback(on_epoch_end=lambda epoch, _: (
    print(f"\nGenerating text after epoch {epoch+1}"),
    print(generate_next(model, "From fairest creatures we desire increase,".lower(), temperature=0.8))
))

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=1, min_lr=1e-4
)
