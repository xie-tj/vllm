# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position acceptance estimates, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification is supported for model-based drafters (EAGLE, MTP, DFlash, DSpark and standalone draft models). The per-position acceptance estimates come from one of two sources:

- **Confidence head**: DSpark checkpoints that ship a trained confidence head use its predictions directly.
- **Online acceptance estimator**: every other drafter predicts acceptance from its draft logits with a small logistic model, recalibrated during serving against the target's verification results.

## Usage

It is off by default. Enable it in the speculative config with `enable_adaptive_verification`. For example, with DSpark:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Or with DFlash on MiMo-V2.5-Pro, whose drafter ships in the checkpoint's `dflash/` subdirectory:

```bash
hf download XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash --local-dir MiMo-V2.5-Pro-FP4-DFlash

vllm serve MiMo-V2.5-Pro-FP4-DFlash \
  --tensor-parallel-size 4 --trust-remote-code --max-model-len auto \
  --speculative-config '{
    "method": "dflash",
    "model": "MiMo-V2.5-Pro-FP4-DFlash/dflash",
    "num_speculative_tokens": 8,
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).
- Without a confidence head, `use_local_argmax_reduction` is rejected. The estimator needs the full draft logits, which local argmax reduction never materializes.
- Automatic backend selection excludes FlashAttention 2 for adaptive verification
  because its FULL graphs require uniform query lengths. Compatible backends
  keep their normal priority; explicit incompatible choices are still rejected.

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```

### Online cost feedback

`enable_online_cost_feedback` updates the cost estimate from completed serving
steps as workload costs drift from the startup profile. It is disabled by
default and requires `enable_adaptive_verification: true`. For an existing MTP
configuration, enable it with:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 3,
  "enable_adaptive_verification": true,
  "enable_online_cost_feedback": true
}
```

The feedback uses a DeepInfer-style robust estimator. Its captured forward
curve is measured separately with no prior KV history, using five startup
measurements per FULL physical token size. Shapes are measured largest first,
with one untimed warmup replay followed by five timed forward-only replays.
A device-group reduction after every replay stays outside the timing events.
All target, inner, draft and profiling graph capture finishes before this
measurement phase; timed samples never interleave with graph capture or Python
model forwards. It discards the
minimum and maximum, and averages the remaining three. The raw curve is not
made monotonic: candidates use the next captured size's original cost, even
when a larger graph measured faster. Runtime residuals use this same raw
baseline. Feedback-disabled and unsupported paths retain vLLM's existing
median-based, monotonic startup tables.

Online feedback adds a non-negative correction per physical token to that raw
forward curve, then adds a runtime sampler estimate. It changes costs for the
existing per-request
verification budget, rather than introducing a separate batch-level draft-length
policy.

Here, **sampler** means the entire real `sample_tokens` stage, starting immediately
before target logits and verification/sampling. It includes state updates,
`prepare_propose`, MTP or other draft proposal, confidence publication, draft
handling and KV finalization before the function returns. Its completion envelope
uses independent per-step timing markers on the existing main, output-copy,
confidence-copy and, for structured outputs, draft-copy streams. Once all markers
have completed, sampler time is the maximum elapsed time from sampler start to
each completion marker. Overlapping copies and draft execution are not added
together. Measurement creates no extra streams or stream dependencies, so it
does not introduce cross-step FIFO waits or new inference waits. The markers
are separate from the copy pipelines' reusable synchronization events.
This measures only those dependencies,
not HTTP end-to-end latency, CPU response processing, or independent KV connector
and other background work that has not joined the measured streams.

For FULL-graph costs, the sampler estimate uses the latest nine completed samples
for the current request count, once that bucket has at least five samples. A
sparse or unseen bucket falls back to the median of the latest nine samples
across request counts, also requiring five samples. Before that, the estimate is
zero: FULL-graph costs do not fall back to the startup draft curve. Eager,
prefill and feedback-disabled paths retain the existing startup costs.
For pure decode, if the minimum non-draft token count fits a captured graph,
online feedback considers only verification budgets up to the capture limit.
Larger eager-path candidates are excluded instead of mixing their old costs with
the new full-sampler costs. If even the minimum token count exceeds that limit,
or the batch includes prefill, the entire decision retains the existing startup
costs.

Forward corrections use at most 5,000 completed samples. Each physical-token-size
bucket needs ten samples before its median per-token residual contributes to the
median slope; the correction is zero before any bucket is ready and is clamped to be
non-negative. Exact forward medians are maintained incrementally over the bounded
window, without a new package dependency. A cold sampler bucket can therefore
use the global estimate while its forward correction is still zero.

Tensor parallelism is supported: TP rank 0 collects runtime samples, maintains
the cost model and broadcasts the chosen budget. When the budget trims drafts,
the admitted per-request counts are also broadcast before building the batch
layout. This keeps execution shapes consistent despite rank-local timing or
confidence differences. Other TP ranks do not run independent calibrators.
Budget and layout consensus also applies when cost feedback is disabled.

Dense data-parallel replicas learn independently within their own TP groups.
For example, `--tensor-parallel-size 2 --data-parallel-size 4` uses eight GPUs
with four independent cost models.

MoE data parallelism uses one cost model for the TP×DP
expert execution group. Each DP replica contributes its expected accepted tokens;
the group chooses the complete vector of replica budgets against their
common physical graph size, including padding imposed by idle replicas. Its
objective is total expected acceptance divided by the cooperating forward cost
plus a driver sampler estimate indexed by the largest real replica request count,
the agreed batch count used by vLLM's cooperating MTP decode. Local request counts
remain in observation metadata; they are not summed into a global sampler.
Only the group driver records runtime observations. Startup forward curves use
the group mean of rank-local trimmed measurements, with valid dummy tokens and
deterministic physical expert selections following the GPU random-routing
protocol in DeepInfer's pinned DeepGEMM
[`store_random_topk_token`](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh#L219).
Philox4x32-10 uses the rank as seed and `token * 32 + lane` as subsequence;
the first two words supply a positive uniform weight and a draw without
replacement. Weights are not normalized. These tables include redundant physical
experts and bypass trained score biases, hash choices and logical-to-physical
EPLB remapping. Local EP streams use their EP rank; gathered tensor-parallel
streams use a common seed so all shards execute the same physical selections.
The native shared-expert slot IDs, gate logits and gate weights are retained.

Tables are materialized on the host before capture and copied to the device;
there is no RNG kernel or RNG-state advancement in the measured replay. Startup
uses startup FULL graphs in a separate shared pool. Profiling routing is removed
before each serving graph is captured; capture and measurement preserve the
serving RNG state. No random-routing operations are added to serving graphs.
Startup graph owners are retained as serving-capture dependencies: native backend
caches may borrow resources recorded during startup. They are not replayed while
serving, and are released together with the serving graphs. Their retained memory
is included in actual capture memory usage; the preliminary graph-memory estimate
can be lower than this measured usage.
The selected expert backend and its settings remain unchanged. Kernels without
a native precomputed physical-routing entry (including monolithic logits-only
kernels and virtual zero-expert routers) reject this profiling protocol explicitly
instead of substituting a different backend or approximating random routing.
This preserves the routing-load protocol, rather than reproducing DeepGEMM's
fused random-generation kernel cost on a different backend.

Native fallback profiling retains its
configured KV context and routing behavior. Startup measurements do not record
synthetic expert load into EPLB serving statistics.

With DP size one, including EP configurations, the existing TP driver and TP
budget/layout consensus are used directly. No DP budget coordinator or per-step
metadata/price reduction is created for that path.
The joint coordinator's fixed-proposal and expert-backend restrictions do not
apply to DP=1. Native backend, graph and parallel lifecycle checks still apply;
this does not establish support for online elastic reconfiguration.
For a dynamic proposal with fewer steps than the configured maximum, the entire
budget decision uses native startup prices. Runtime calibration remains limited
to full-step proposals, so their sampler estimates cannot price shorter proposals.

For MoE DP size greater than one, all ranks, including idle ranks, participate
in one metadata/price reduction per decision. The driver alone learns runtime
costs. Every rank derives identical
budgets from the same published prices and acceptance data, without an extra
object broadcast. Only DP leaders copy confidence histories to the host and
contribute acceptance vectors; TP followers retain GPU confidence updates and
metadata votes. Leaders consume only completed host histories: if the next
confidence copy is still in flight, they keep the previous history and omit a
new host copy for that step. GPU ranking confidences are still updated every
step, and request-slot resets are applied when the completed history is adopted.
DP=1 and feedback-disabled history publication retain their existing behavior.
Startup fallback budgets retain TP consensus. Sampler price
tables are reused until observations
change. Collected observations enter the following step's prices.
Graph dispatch and DP padding use vLLM's native coordination.
Decisions and observations carry a step, cost version, profiler epoch, replica
request counts and physical shapes. If any rank is profiling or a replica has
prefill, the entire group falls back to startup costs. The driver does not learn
from dummy sampler work. The same 9/5 sampler rule applies to the cooperating
draft batch; this remains a driver estimate, not an all-rank latency average.
Worker `get_online_cost_feedback_stats` collective RPC returns bounded observation
and admitted-budget diagnostics for verification and performance analysis.

For MoE DP size greater than one, online feedback requires static groups and,
when EP is enabled, the `allgather_reducescatter` backend. Batch-size-dependent
draft-step schedules remain unsupported for this joint coordinator: the proposal
step count must stay fixed across cooperating replicas. Context parallelism and
dual-batch overlap remain unsupported by online feedback at every DP size.
It learns only from real pure-decode steps using FULL CUDA graphs; prefill,
dummy steps and unsupported execution shapes retain startup costs. Existing
adaptive-verification restrictions on attention backends, LoRA and pipeline
parallelism still apply.

Every eligible step attempts a timing sample. The pending queue holds at most
16 samples; when it is full, new samples are skipped rather than blocking, so
this is not a guarantee that every step is retained. Completed measurements are
collected later with non-blocking event queries. Feedback does not wait for a
running sampler or draft to finish, and async scheduling remains supported.
vLLM-managed profiler sessions are excluded from feedback: their timing overhead
must not become the serving cost estimate.
When profiling starts or stops, active and pending feedback samples are discarded
so measurements spanning a profiler session cannot enter the cost model. Pending
GPU work still completes before its timing markers are released.
