import tensorflow as tf
from load_data import prep_data
from LR_schedule import CustomSchedule
from util import masked_loss, masked_accuracy, BUFFER_SIZE, BATCH_SIZE, d_model
from transformer_init import transformer

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
checkpoint_path = "training_1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_freq='epoch',
                                                 verbose=1)

dataset_path = 'my_uk-en_dataset'
train_batches, val_batches = prep_data(dataset_path, BUFFER_SIZE, BATCH_SIZE)
transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)
