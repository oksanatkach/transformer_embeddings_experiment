import tensorflow as tf
from load_data import prep_data
from util import masked_loss, masked_accuracy, BUFFER_SIZE, BATCH_SIZE
from LR_schedule import CustomSchedule
from translate import translate

# last_checkpoint_path_model = '../model_after_20_epochs.ckpt'
last_checkpoint_path_weights = "../training_2/cp-0020.ckpt"

transformer = tf.keras.models.load_model(last_checkpoint_path_weights, custom_objects={'masked_accuracy': masked_accuracy,
                                                                   'masked_loss': masked_loss,
                                                                   'CustomSchedule': CustomSchedule
                                                                 })
# translate(transformer, "Hi", "Привіт")

dataset_path = '../my_uk-en_dataset'
train_batches, val_batches = prep_data(dataset_path, BUFFER_SIZE, BATCH_SIZE)

checkpoint_path = "training_3/{epoch:02d}-{val_masked_accuracy:.2f}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_freq='epoch',
                                                 monitor='val_masked_accuracy',
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 mode='max')

transformer.fit(train_batches,
                epochs=10,
                validation_data=val_batches,
                callbacks=[cp_callback])
