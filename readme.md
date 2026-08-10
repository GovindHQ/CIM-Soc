# cim_sim — 32×32 CIM array model for ViT attention

A software model of a single 32×32 weight-stationary compute-in-memory array,
and a drop-in replacement for the matmuls inside TinyViT-5M attention.

Scope of this version, as scoped: **one array, one weight plane, INT4 weights
and activations, no ADC model.** The three attention matmul sites — fused QKV
projection, QKᵀ, AV — execute on the array. LayerNorm, attention biases, softmax,
the output projection and the MLP stay in float, which is where they stay in the
hardware too.

## Layout

```
cim_sim/
  quant.py       symmetric integer quantization; the scale-placement argument
  array.py       the 32x32 primitive + activity counters
  matmul.py      the scheduler: depth blocks, output blocks, token blocking
  attention.py   CIMAttention (torch nn.Module) + model surgery helpers
  report.py      tables
verify_array.py  correctness harness — run this first
run_tinyvit.py   pretrained TinyViT-5M end to end
tiny_vit.py      unmodified upstream model source (microsoft/Cream, MIT)
_shim/timm/      three-symbol timm stand-in so tiny_vit.py imports cleanly
```

## Run

```bash
pip install torch numpy
python verify_array.py                     # prove the schedule is exact
python run_tinyvit.py --stage3-only        # one stage-3 block, full tile map
python run_tinyvit.py --ablate             # which matmul site costs accuracy
python run_tinyvit.py --sweep-bits         # 2/3/4/6/8 bit
python run_tinyvit.py --sweep-token-block  # writes vs psum storage
python run_tinyvit.py                      # full model
```

Weights download automatically on first run (~22 MB).

## The scheduler

`cim_matmul(X[N,K], W[K,M])` is the only entry point. It pads K to a multiple of
32 (depth blocks) and M to a multiple of 32 (output blocks) — that padding *is*
the idle rows and idle columns of the real array — then walks:

```
for token block tb in ceil(N/T):
    for output block ob in ceil(M/32):
        for depth block db in ceil(K/32):
            load tile W[db, ob]              <- one analog write (single plane)
            acc[tb, ob] += array.mvm(Xq[tb, db])
```

Weight writes = `ceil(N/T) · ceil(M/32) · ceil(K/32)`. Live partial sums = `T·32`
words. `T` is the only knob and it trades those two against each other with
**bit-identical results**; `verify_array.py` asserts this.

## Why the scales are placed where they are

Accumulation across depth blocks happens on raw integers, so a single rescale at
the end is only valid if the scale is constant along the contraction axis. Hence
one activation scale per token (shared across all K) and one weight scale per
output column (shared across all K). A per-depth-block scale would be more
accurate but forces a rescale-and-requantize between every depth block, which
defeats the cheap integer accumulator. This is the main place where the
quantization scheme is constrained by the dataflow rather than by accuracy.

Post-softmax attention probabilities are quantized **unsigned** (0…15 rather than
−8…7), since they have no negative side and a sign bit would be wasted.

## Extension points

Each of these is a small change in one file:

- **ADC model** — quantize the return value of `CIMArray.mvm` before it leaves
  the array. `max_abs_colsum` in the stats already reports the required input
  range (12 bits signed on the full model).
- **Four resident planes** — `load_weight_tile` already takes a `tile_id`; hold a
  dict of up to four resident tiles and only count a write on eviction.
- **Multiple arrays** — spatially unroll the `db` loop across arrays and reduce
  digitally; the counters divide, the numerics do not change.
- **8-bit via bit-slicing** — two 4-bit passes per operand with a shift-add in
  the digital accumulator; slots in at `cim_matmul` between quantize and pad.
- **Convolution** — im2col the activations, then call `cim_matmul` unchanged. The
  scheduler does not care whether a row is a token or an output pixel.
