import tensorflow as tf
from util import masked_loss, masked_accuracy, BUFFER_SIZE, BATCH_SIZE, d_model
from LR_schedule import CustomSchedule
from transformer_init import transformer
from load_data import prep_data
# tf.compat.v1.disable_eager_execution()
filepath = '../model_after_20_epochs.ckpt'
# load the model
new_transformer = tf.keras.models.load_model(filepath, custom_objects={'masked_accuracy': masked_accuracy,
                                                                   'masked_loss': masked_loss,
                                                                   'CustomSchedule': CustomSchedule
                                                             })

# learning_rate = CustomSchedule(d_model, stopped_at=20)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# transformer.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy])
#
# new_transformer.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy])

new_transformer.load_weights(filepath)
# print(new_transformer.optimizer)

dataset_path = '../my_uk-en_dataset'
train_batches, val_batches = prep_data(dataset_path, BUFFER_SIZE, BATCH_SIZE)
# print(train_batches.take(1)))
# for train, label in train_batches.take(1):
#     transformer(train)

# new_transformer.optimizer.build(transformer.trainable_variables)
#
#
# print(transformer.optimizer)
# print([v.name for v in transformer.optimizer.variables()])      # ['iteration:0', 'Adam/m/dense/kernel:0', 'Adam/v/dense/kernel:0', 'Adam/m/dense/bias:0', 'Adam/v/dense/bias:0']
# print([v.name for v in new_transformer.optimizer.variables()])  # ['iteration:0', 'm/dense_2/kernel:0', 'v/dense_2/kernel:0', 'm/dense_2/bias:0', 'v/dense_2/bias:0']

checkpoint_path = "training_3/{epoch:02d}-{val_masked_accuracy:.2f}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_freq='epoch',
                                                 monitor='val_masked_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='max')

new_transformer.fit(train_batches,
                epochs=10,
                validation_data=val_batches,
                callbacks=[cp_callback])
