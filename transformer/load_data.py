import tensorflow as tf
from util import tokenizers, MAX_TOKENS


def prepare_batch(en, uk):
    with tf.device('/cpu:0'):
        en = tokenizers.en.tokenize(en)  # Output is ragged.
        uk = tokenizers.uk.tokenize(uk)

    en = en[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    en = en.to_tensor()  # Convert to 0-padded dense Tensor

    uk = uk[:, :(MAX_TOKENS + 1)]
    uk_inputs = uk[:, :-1].to_tensor()  # Drop the [END] tokens
    uk_labels = uk[:, 1:].to_tensor()  # Drop the [START] tokens

    return (en, uk_inputs), uk_labels


def make_batches(ds, BUFFER_SIZE, BATCH_SIZE):
    return (ds.shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))


def prep_data(dataset_path, BUFFER_SIZE, BATCH_SIZE):
    dataset = tf.data.experimental.load(dataset_path)

    DATASET_SIZE = dataset.cardinality().numpy()
    train_size = int(0.8 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Create training and validation set batches.
    train_batches = make_batches(train_dataset, BUFFER_SIZE, BATCH_SIZE)
    val_batches = make_batches(val_dataset, BUFFER_SIZE, BATCH_SIZE)

    return train_batches, val_batches
