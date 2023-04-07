import tensorflow as tf
from load_data import prep_data
from util import masked_loss, masked_accuracy, BUFFER_SIZE, BATCH_SIZE
from LR_schedule import CustomSchedule
from util import d_model
import pickle
import keras.backend as K

filepath = 'model_after_20_epochs.ckpt'
# load the model
transformer = tf.keras.models.load_model(filepath, custom_objects={'masked_accuracy': masked_accuracy,
                                                                   'masked_loss': masked_loss,
                                                                   'CustomSchedule': CustomSchedule
                                                                 })

learning_rate = CustomSchedule(d_model, stopped_at=20)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=None)

# Initialise dictionaries to store the training and validation losses
train_loss_dict = {}
val_loss_dict = {}

# Include metrics monitoring
train_loss = tf.metrics.Mean(name='train_loss')
train_accuracy = tf.metrics.Mean(name='train_accuracy')
val_loss = tf.metrics.Mean(name='val_loss')


# Speeding up the training process
@tf.function
def train_step(encoder_input, decoder_input, decoder_output):
    with tf.GradientTape() as tape:

        # Run the forward pass of the model to generate a prediction
        prediction = transformer((encoder_input, decoder_input), training=False)

        # Compute the training loss
        loss = masked_loss(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = masked_accuracy(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, transformer.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, transformer.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)


dataset_path = '../my_uk-en_dataset'
train_batches, val_batches = prep_data(dataset_path, BUFFER_SIZE, BATCH_SIZE)
epochs = 10

for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    print("\nStart of epoch %d" % (epoch + 1))

    # Iterate over the dataset batches
    for step, (train_batchX, train_batchY) in enumerate(train_batches):
        encoder_input, decoder_input = train_batchX
        decoder_output = train_batchY

        # Define the encoder and decoder inputs, and the decoder output
        # encoder_input = train_batchX[:, 1:]
        # decoder_input = train_batchY[:, :-1]
        # decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        if step % 50 == 0:
            print(f"Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} "
                  + f"Accuracy {train_accuracy.result():.4f}")

    # Run a validation step after every epoch of training
    for val_batchX, val_batchY in val_batches:
        # Define the encoder and decoder inputs, and the decoder output
        encoder_input = val_batchX[:, 1:]
        decoder_input = val_batchY[:, :-1]
        decoder_output = val_batchY[:, 1:]

        # Generate a prediction
        prediction = transformer((encoder_input, decoder_input), training=False)

        # Compute the validation loss
        loss = masked_loss(decoder_output, prediction)
        val_loss(loss)

    # Print epoch number and accuracy and loss values at the end of every epoch
    print(f"Epoch {epoch+1}: Training Loss {train_loss.result():.4f}, "
          + f"Training Accuracy {train_accuracy.result():.4f}, "
          + f"Validation Loss {val_loss.result():.4f}")

    # Save a checkpoint after every epoch
    if (epoch + 1) % 1 == 0:
        save_path = ckpt_manager.save()
        print(f"Saved checkpoint at epoch {epoch+1}")

        # Save the trained model weights
        transformer.save_weights("weights/wghts" + str(epoch + 1) + ".ckpt")

        # save optimizer state
        symbolic_weights = getattr(transformer.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('optimizer_states/optimizer' + str(epoch + 1) + '.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

        train_loss_dict[epoch] = train_loss.result()
        val_loss_dict[epoch] = val_loss.result()

# Save the training loss values
with open('./train_loss.pkl', 'wb') as file:
    pickle.dump(train_loss_dict, file)

# Save the validation loss values
with open('./val_loss.pkl', 'wb') as file:
    pickle.dump(val_loss_dict, file)
