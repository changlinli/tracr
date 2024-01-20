# %%
from typing import Set
from tracr.rasp import rasp
import einops
import torch
import numpy as np
import jax.numpy as jnp
from transformer_lens import HookedTransformer, HookedTransformerConfig

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

length = make_length()  # `length` is not a primitive in our implementation.
opp_index = length - rasp.indices - 1
flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
reverse = rasp.Aggregate(flip, rasp.tokens)
reverse_vocab = {1, 2, 3}

from tracr.compiler import compiling

bos = "BOS"
model = compiling.compile_rasp_to_model(
    reverse,
    vocab={1, 2, 3},
    max_seq_len=5,
    compiler_bos=bos,
)
print(model.apply(["BOS", 1, 2, 3, 1, 2]).decoded)

# %%
def make_frac_prevs(bools: rasp.SOp) -> rasp.SOp:
  """Count the fraction of previous tokens where a specific condition was True.

   (As implemented in the RASP paper.)

  Example usage:
    num_l = make_frac_prevs(rasp.tokens=="l")
    num_l("hello")
    >> [0, 0, 1/3, 1/2, 2/5]

  Args:
    bools: SOp mapping a sequence to a sequence of booleans.

  Returns:
    frac_prevs: SOp mapping an input to a sequence, where every element
      is the fraction of previous "True" tokens.
  """
  bools = rasp.numerical(bools)
  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
  return rasp.numerical(rasp.Aggregate(prevs, bools,
                                       default=0)).named("frac_prevs")

def make_pair_balance(sop: rasp.SOp, open_token: str,
                      close_token: str) -> rasp.SOp:
  """Return fraction of previous open tokens minus the fraction of close tokens.

   (As implemented in the RASP paper.)

  If the outputs are always non-negative and end in 0, that implies the input
  has balanced parentheses.

  Example usage:
    num_l = make_pair_balance(rasp.tokens, "(", ")")
    num_l("a()b(c))")
    >> [0, 1/2, 0, 0, 1/5, 1/6, 0, -1/8]

  Args:
    sop: Input SOp.
    open_token: Token that counts positive.
    close_token: Token that counts negative.

  Returns:
    pair_balance: SOp mapping an input to a sequence, where every element
      is the fraction of previous open tokens minus previous close tokens.
  """
  bools_open = rasp.numerical(sop == open_token).named("bools_open")
  opens = rasp.numerical(make_frac_prevs(bools_open)).named("opens")

  bools_close = rasp.numerical(sop == close_token).named("bools_close")
  closes = rasp.numerical(make_frac_prevs(bools_close)).named("closes")

  pair_balance = rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1))
  return pair_balance.named("pair_balance")

print(f'{make_pair_balance(rasp.tokens, "(", ")")("(ab)(bc)((((((ab))))))")=}')

# %%

def frac_prevs(sop: rasp.SOp, val: rasp.Value) -> rasp.SOp:

  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
  return rasp.Aggregate(
    prevs,
    sop == val
  )

def pair_balance(open: rasp.Value, close: rasp.Value) -> rasp.SOp:
  opens = frac_prevs(rasp.tokens, open)
  closes = frac_prevs(rasp.tokens, close)
  return opens - closes


print(f"{pair_balance('(', ')')('((ab))(()(c))')=}")

print(f'{frac_prevs(rasp.tokens, "a")("aaabbbbaaa")=}')


# %%

program = pair_balance('(', ')')
program_vocab = {'a', 'b', 'c', 'd', 'e', 'f', '(', ')'}
assembled_model = compiling.compile_rasp_to_model(
  program, 
  vocab=program_vocab,
  max_seq_len=10,
  compiler_bos="BOS",
  compiler_pad="PAD",
  mlp_exactness=100,
)

print(f"{assembled_model.apply(['BOS', 'a', 'c', '(', ')']).decoded=}")

# %%

print(f"{assembled_model.model_config=}")

for output in assembled_model.apply(['BOS', 'a', 'b', 'c','d']).layer_outputs:
   print(f"{output.shape=}")


print(f'{model.params["token_embed"]["embeddings"].shape=}')

d_model = model.params["token_embed"]['embeddings'].shape[1]

# %%

# Mostly copied from https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb
def compile_rasp_to_transformer_lens(
  program: rasp.SOp,
  vocab: Set[rasp.Value],
  max_seq_len: int,
  causal: bool = False,
  compiler_bos: str = compiling.COMPILER_BOS,
  compiler_pad: str = compiling.COMPILER_PAD,
  mlp_exactness: int = 100,
) -> HookedTransformer:
  jax_model = compiling.compile_rasp_to_model(
     program=program,
     vocab=vocab,
     max_seq_len=max_seq_len,
     causal=causal,
     compiler_bos=compiler_bos,
     compiler_pad=compiler_pad,
     mlp_exactness=mlp_exactness,
  )

  n_heads = jax_model.model_config.num_heads
  n_layers = jax_model.model_config.num_layers
  d_head = jax_model.model_config.key_size
  d_mlp = jax_model.model_config.mlp_hidden_size
  act_fn = "relu"
  normalization_type = "LN"  if jax_model.model_config.layer_norm else None
  attention_type = "causal"  if jax_model.model_config.causal else "bidirectional"
  n_ctx = jax_model.params["pos_embed"]['embeddings'].shape[0]
  # Equivalent to length of vocab, with BOS and PAD at the end
  d_vocab = jax_model.params["token_embed"]['embeddings'].shape[0]
  # Residual stream width, I don't know of an easy way to infer it from the above config.
  # CL: Note that unlike standard transformers, d_model does *not* have to be
  # d_head * n_heads In particular the output matrix (i.e.
  # transformer/layer_n/attn/linear) can reshape the output.
  d_model = jax_model.params["token_embed"]['embeddings'].shape[1]

  # Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
  # In practice, we always feed the logits into an argmax
  d_vocab_out = jax_model.params["token_embed"]['embeddings'].shape[0] - 2

  print(f'{jax_model.params["token_embed"]["embeddings"].shape=}')
  print(f'{jax_model.model_config=}')
  sd = {}
  sd["pos_embed.W_pos"] = jax_model.params["pos_embed"]['embeddings']
  sd["embed.W_E"] = jax_model.params["token_embed"]['embeddings']
  # Equivalent to max_seq_len plus one, for the BOS

  # The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
  # This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
  sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

  for l in range(n_layers):
      sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/key"]["w"],
          "d_model (n_heads d_head) -> n_heads d_model d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/key"]["b"],
          "(n_heads d_head) -> n_heads d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/query"]["w"],
          "d_model (n_heads d_head) -> n_heads d_model d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/query"]["b"],
          "(n_heads d_head) -> n_heads d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/value"]["w"],
          "d_model (n_heads d_head) -> n_heads d_model d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/value"]["b"],
          "(n_heads d_head) -> n_heads d_head",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
          jax_model.params[f"transformer/layer_{l}/attn/linear"]["w"],
          "(n_heads d_head) d_model -> n_heads d_head d_model",
          d_head = d_head,
          n_heads = n_heads
      )
      sd[f"blocks.{l}.attn.b_O"] = jax_model.params[f"transformer/layer_{l}/attn/linear"]["b"]

      sd[f"blocks.{l}.mlp.W_in"] = jax_model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
      sd[f"blocks.{l}.mlp.b_in"] = jax_model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
      sd[f"blocks.{l}.mlp.W_out"] = jax_model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
      sd[f"blocks.{l}.mlp.b_out"] = jax_model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]

  for k, v in sd.items():
      # I cannot figure out a neater way to go from a Jax array to a numpy array lol
      sd[k] = torch.tensor(np.array(v))

  cfg = HookedTransformerConfig(
      n_layers=n_layers,
      d_model=d_model,
      d_head=d_head,
      n_ctx=n_ctx,
      d_vocab=d_vocab,
      d_vocab_out=d_vocab_out,
      d_mlp=d_mlp,
      n_heads=n_heads,
      act_fn=act_fn,
      attention_dir=attention_type,
      normalization_type=normalization_type,
  )
  transformer_lens_model = HookedTransformer(cfg)
  transformer_lens_model.load_state_dict(sd, strict=False)
  return transformer_lens_model

transformer_lens_model = compile_rasp_to_transformer_lens(
  reverse, 
  vocab=reverse_vocab,
  max_seq_len=10,
  compiler_bos="BOS",
  compiler_pad="PAD",
  mlp_exactness=100,
)

standard_jax_model = compiling.compile_rasp_to_model(
  reverse, 
  vocab=reverse_vocab,
  max_seq_len=10,
  compiler_bos="BOS",
  compiler_pad="PAD",
  mlp_exactness=100,
)


# %%

INPUT_ENCODER = standard_jax_model.input_encoder
OUTPUT_ENCODER = standard_jax_model.output_encoder

def create_model_input(input, input_encoder=INPUT_ENCODER):
    encoding = input_encoder.encode(input)
    return torch.tensor(encoding).unsqueeze(dim=0)

def decode_model_output(logits, output_encoder=OUTPUT_ENCODER, bos_token=INPUT_ENCODER.bos_token):
  max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
  decoded_output = output_encoder.decode(max_output_indices.tolist())
  decoded_output_with_bos = [bos_token] + decoded_output[1:]
  return decoded_output_with_bos

# %%
input = [bos, 1, 2, 3]
out = standard_jax_model.apply(input)
print("Original Decoding:", out.decoded)

input_tokens = standard_jax_model.input_encoder.encode(input)
out_manual = standard_jax_model.forward(standard_jax_model.params, jnp.array([input_tokens]))
print("Original Decoding Manually written:", out_manual.unembedded_output)

program(input)

input_tokens_tensor = create_model_input(input)
logits = transformer_lens_model(input_tokens_tensor)
decoded_output = decode_model_output(logits)
print("TransformerLens Replicated Decoding:", decoded_output)
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

out = model.apply([bos, 1, 2, 3])
print(f"{out.decoded=}")

# %%


compile_rasp_to_transformer_lens(
  reverse,
  vocab={1, 2, 3},
  max_seq_len=5,
  compiler_bos=bos,
)

# %%
print(f"{model.model_config=}")
model.params[f"transformer/layer_0/attn/linear"]['w'].shape