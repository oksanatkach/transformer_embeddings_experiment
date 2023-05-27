import tensorflow as tf
from translate import translate
from transformer_init import transformer
from util import tokenizers
from translator import Translator

model_path = '/Users/oksanatkach/PycharmProjects/transformer_embeddings_experiments/best_model'

transformer.load_weights(model_path)
# translate(transformer, "Hi", "Привіт")


def prep_en_sentence(sentence):
    sentence = tf.constant([sentence])
    with tf.device('/cpu:0'):
        return tokenizers.en.tokenize(sentence).to_tensor()


def prep_uk_sentence(sentence):
    sentence = tf.constant([sentence])
    with tf.device('/cpu:0'):
        return tokenizers.uk.tokenize(sentence).to_tensor()


source_sent = 'Tap here to continue'
target_sent = 'Натисніть тут, щоб продовжити'
glossary_term = ('tap', 'торкніться')
source_input_vector = prep_en_sentence(source_sent)
# print(source_input_vector)
target_input_vector = prep_uk_sentence(target_sent)
# print(target_input_vector)

source_term_vector = prep_en_sentence(glossary_term[0])
# print(source_term_vector)
target_term_vector = prep_uk_sentence(glossary_term[1])
# print(target_term_vector)

source_random_sent = 'random stuff'
source_random_vector = prep_en_sentence(source_random_sent)
target_random_sent = 'випадкове речення'
target_random_vector = prep_en_sentence(target_random_sent)

source_main_processed, target_main_processed = transformer.get_final_vectors(source_input_vector, target_input_vector)
source_term_processed, target_term_processed = transformer.get_final_vectors(source_term_vector, target_term_vector)
source_random_processed, target_random_processed = transformer.get_final_vectors(source_random_vector, target_random_vector)

source_term_processed = tf.squeeze(source_term_processed, 0)
# print(source_term_processed)
source_term_processed = tf.math.reduce_mean(source_term_processed, 0)
# print(source_term_processed)

target_term_processed = tf.squeeze(target_term_processed, 0)
target_term_processed = tf.math.reduce_mean(target_term_processed, 0)

source_random_processed = tf.squeeze(source_random_processed, 0)
source_random_processed = tf.math.reduce_mean(source_random_processed, 0)

target_random_processed = tf.squeeze(target_random_processed, 0)
target_random_processed = tf.math.reduce_mean(target_random_processed, 0)

# print('Tap')
# source_term_main = tf.expand_dims(source_main_processed[0][1], 0)
source_term_main = source_main_processed[0][1]
print(source_term_main)
# print('####################')
# print('Натисніть')
target_term_main = tf.expand_dims(target_main_processed[0][1:4], 0)
target_term_main = tf.math.reduce_mean(target_term_main, 1)
# print('####################')
# print('Tap')
# print(source_term[0][1])
# print('####################')
# print('Tоркніться')
# print(target_term[0][1])
# print(target_term[0][2])
# print(target_term[0][3])
# print(target_term[0][4])

print('main sent term to singular term source')
print(tf.tensordot(source_term_main, source_term_processed, 1))
# print(tf.matmul(source_term_main, tf.transpose(source_term_processed)))

print('main sent term to singular term target')
print(tf.tensordot(target_term_main, target_term_processed, 1))
# print(tf.matmul(target_term_main, tf.transpose(target_term_processed)))
print('main sent term to random source')
print(tf.tensordot(source_term_main, source_random_processed, 1))
# print(tf.matmul(source_main_processed, tf.transpose(source_random_processed)))
print('main sent term to random target')
print(tf.tensordot(target_term_main, target_random_processed, 1))
# print(tf.matmul(target_main_processed, tf.transpose(target_random_processed)))

# lst = [98, 6519, 2100, 125, 12, 120,  122, 4383,  280, 2058]
# lst = [538, 620, 464, 6463]
# for el in lst:
#     input = tf.constant([[el]], dtype=tf.int64)
#     decoded = tokenizers.uk.detokenize(input)
#     print(decoded.numpy()[0].decode('utf8'))
# 98, 6519, 2100
# 1,    2,    3
# 538, 620, 464, 6463
# 1,    2,   3,   4
