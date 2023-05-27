import tensorflow as tf
# unused import to make tf load the model properly
import tensorflow_text as text

BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_TOKENS = 128
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
tokenizer_path = '/Users/oksanatkach/PycharmProjects/transformer_embeddings_experiments/ted_hrlr_translate_uk_en_converter'
tokenizers = tf.saved_model.load(tokenizer_path)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)
