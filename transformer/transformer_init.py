from transformer import Transformer
from util import tokenizers, num_layers, d_model, num_heads, dff, dropout_rate

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.uk.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)
