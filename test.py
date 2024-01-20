# %%
from tracr.rasp import rasp

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

length = make_length()  # `length` is not a primitive in our implementation.
opp_index = length - rasp.indices - 1
flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
reverse = rasp.Aggregate(flip, rasp.tokens)

from tracr.compiler import compiling

bos = "BOS"
model = compiling.compile_rasp_to_model(
    reverse,
    vocab={1, 2, 3},
    max_seq_len=5,
    compiler_bos=bos,
)
print(model.apply(["BOS", 1, 2, 3, 1, 2]).decoded)
