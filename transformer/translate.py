from translator import Translator
from util import tokenizers


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


def translate(transformer, sentence, ground_truth):
    translator = Translator(tokenizers, transformer)
    # translated_text, translated_tokens, attention_weights = translator(sentence)
    translated_text, translated_tokens = translator(sentence)
    print_translation(sentence, translated_text, ground_truth)
