# The Logic Chain of LLM Infra

<p class="lc-lead">This post is for readers <b>who care about LLM system design and want an overall picture</b>. Starting from the LLM itself, it follows one causal chain through the main designs of training and inference systems, with the focus on the motivation of each design. Each topic covers, in order, <b>what problem arises</b>, <b>how it is solved</b>, and <b>what is improved and what is sacrificed</b>. Only the principles and numbers needed to understand the designs are kept; implementation details are left out. All numbers come from public hardware specifications and papers, with the sources listed at the end.</p>

<div class="lc-map" markdown="0">
<div class="lc-row">
<div class="lc-card">
<div class="lc-head"><h2 id="llm">LLM</h2></div>
<dl class="lc-fact">
<dt>Scope</dt><dd><p>The training and inference systems of LLMs (large language models), and where their main designs come from.</p></dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">To understand an LLM, first look at its basic architecture</p>
<ol class="lc-steps">
<li>This post is about the training and inference systems of LLMs.</li>
<li>The model architecture determines the system's costs: analyzing the costs requires knowing the architecture first.</li>
<li>Almost all current LLMs use the same architecture: the Transformer.</li>
<li>So the analysis starts with the architecture of the Transformer and the data on the GPU during training.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="transformer">Basic Architecture: Transformer</h2></div>
<dl class="lc-fact">
<dt>Architecture</dt><dd><p>Each layer has two main modules: <b>attention</b> (lets each token gather information from the tokens before it; split internally into several heads computed independently) and <b>FFN</b> (a two-layer fully connected network, computed for each token separately). $$L$$ layers are stacked, with modules such as tokenization (splitting text into tokens, which are words or parts of words) and embedding added before and after. Most parameters are in the matrices of these $$L$$ layers.</p></dd>
<dt>Data</dt><dd><p>During training, the data on the GPU is of four kinds: <b>parameters</b> (the matrices above, with count $$\Phi$$), <b>gradients</b> and <b>optimizer states</b> (one gradient per parameter, plus two statistics per parameter for Adam). These three together are the training state, and the model alone sets their size. <b>Activations</b> (intermediate results of each module's forward pass, needed in the backward pass) vary in size with the batch (the number of sequences processed at once) and the sequence length. The amount of input data is small, so it can be counted with the activations.</p></dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">This data has only three states, being stored, computed or moved, and each state has its own cost</p>
<ol class="lc-steps">
<li>During training, the data on the GPU is of four kinds: parameters, gradients, optimizer states and activations.</li>
<li>Any piece of data, at any moment, is in one of three states: being stored, being computed on, or being moved.</li>
<li>The three states use three different hardware resources: storing uses memory capacity, computing uses compute, and moving uses bandwidth. Each resource has a limit, usually called the memory wall, the compute wall and the bandwidth wall.</li>
<li>The units of the three walls are Bytes, FLOP/s and Bytes/s.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="costs">Three costs</h2><span class="lc-sub">memory wall · compute wall · bandwidth wall</span></div>
<div class="lc-costs">
<div class="lc-cost mem"><span class="en">Data Placement</span><span class="def"><b>Memory cost</b>: where data is placed and how much GPU memory it takes</span><span class="wall">memory wall · <code>Bytes</code></span><span class="hw">H100: 80 GB</span></div>
<div class="lc-cost calc"><span class="en">Data Calculation</span><span class="def"><b>Compute cost</b>: the floating-point operations performed on data</span><span class="wall">compute wall · <code>FLOP/s</code></span><span class="hw">H100 BF16: 989 TFLOP/s</span></div>
<div class="lc-cost move"><span class="en">Data Movement</span><span class="def"><b>Data-movement cost</b>: moving data between machines, between GPUs, and between HBM and SRAM</span><span class="wall">bandwidth wall · <code>Bytes/s</code></span><span class="hw">H100: HBM 3.35 TB/s, NVLink 450 GB/s, cross-machine network 50 GB/s</span></div>
</div>
<p class="lc-note">The three costs are of different kinds. Memory cost is a capacity: past the memory capacity, the program cannot run. Compute cost and data-movement cost are times, equal to operations ÷ achieved compute and bytes ÷ achieved bandwidth; when compute units sit idle or bandwidth is not fully used, the time grows. HBM is the GPU memory; SRAM is small, fast on-chip storage. H100 figures are for the SXM version: compute is the dense value without sparsity; NVLink and the cross-machine network are counted in one direction, and the cross-machine network assumes one 400 Gb/s NIC per GPU.</p>
<div class="lc-thesis">
<p><b>From here on, what each technique improves and sacrifices is described in terms of these three costs.</b></p>
<p><b>Two questions, in order:</b> ① Is the technique waste removal, that is, does it only remove parts that served no purpose? ② If not, which cost does it improve and which does it sacrifice?</p>
<p><span class="tag tag-waste">waste removal</span> Removes parts that served no purpose, such as duplicated data or idle time of compute units: improves one cost and sacrifices no other; small implementation overheads are not counted.</p>
<p><span class="tag tag-trade">trade-off</span> Improves one cost and sacrifices another. Usually the improved cost is the bottleneck in the current setting, and the sacrificed one has room to spare in that setting.</p>
<p><b>Beyond the three costs:</b> each kernel launch (a kernel is a function run on the GPU) or communication call also has a fixed cost on the order of microseconds, independent of data size; a few techniques sacrifice numerical accuracy, load balance, latency or simplicity of implementation. Cards mark these in gray.</p>
<p class="lc-legend">Colors mark the cost that changes: <span class="q q-mem">memory cost</span> <span class="q q-calc">compute cost</span> <span class="q q-move">data-movement cost</span> <span class="q q-other">beyond the three costs</span></p>
</div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">With the three costs, trade-offs can be analyzed, starting inside one GPU</p>
<div class="lc-formula">\[ T = \frac{6\Phi D}{N \cdot R \cdot \eta} \qquad M = 16\Phi + A \]<p class="fnote">$$T$$: training time; $$D$$: number of training tokens; $$N$$: number of GPUs; $$R$$: achieved compute per GPU (FLOP/s), counting only the model's own operations and not recomputation; $$\eta$$: parallel efficiency. $$M$$: GPU memory used per GPU (byte); $$16\Phi$$ is the training state (parameters, gradients and Adam's two statistics, $$4\Phi$$ bytes each when all are in FP32) and $$A$$ the activations.</p></div>
<ol class="lc-steps">
<li>Training takes about $$6\Phi D$$ FLOP in total: for each parameter and each token, about 2 FLOP in the forward pass and about 4 FLOP in the backward pass.</li>
<li>The model and data set the numerator, so training time can be cut only by increasing the three factors in the denominator: the achieved compute of one GPU $$R$$ (mixed precision, kernel optimization), the number of GPUs $$N$$ (DDP and the parallel methods after it), and the parallel efficiency $$\eta$$ (waiting for communication and idle GPUs keep it below 1). The constraint is that each GPU's memory use $$M$$ does not exceed the memory capacity.</li>
<li>First, two defaults on one GPU are changed, and both changes are trade-offs: giving up a cost with room to spare for the one that is the bottleneck, to raise $$R$$ or lower $$M$$.</li>
<li>The first default is numerical precision: by default each number is stored in FP32, taking 4 bytes, and matrix multiplies do not go through Tensor Cores (when TF32 is not enabled). On H100, the FP32 units reach 67 TFLOP/s and the BF16 Tensor Cores 989 TFLOP/s.</li>
<li>The second is how activations are kept: by default all are saved. A 1.3B model (the GPT-3 XL architecture) processing 32 sequences of 2048 tokens per step has about 110 GB of activations in BF16 without the attention score matrices (about 500 GB with them), while the training state is only 21 GB.</li>
<li>The techniques that change these two defaults are mixed precision and activation checkpointing.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="single-gpu">Trade-offs within one GPU</h2><span class="lc-sub">mixed precision · activation checkpointing</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>With FP32 throughout, the matrix-multiply peak is only about 1/15 of the BF16 Tensor Core peak, and each activation value takes 4 bytes; with all activations saved, they may exceed the memory capacity.</p></dd>
<dt>Solution</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>Mixed precision</b><span class="tag tag-trade">trade-off</span></div>
<p>Matrix-multiply inputs and activations use BF16 (2 bytes per number), with products accumulated in FP32; reductions such as softmax and normalization, and the parameter update, use FP32. The update stays in FP32 because an update smaller than about 1/256 of the parameter is rounded away in BF16. The training state is BF16 parameters and gradients at $$2\Phi$$ each, plus FP32 parameters and Adam's two statistics at $$12\Phi$$, for $$16\Phi$$ bytes in total.</p>
<div class="lc-chips"><span class="q q-calc"><b>Improves</b>compute: about 15 times the matrix-multiply peak</span><span class="q q-mem"><b>Improves</b>memory: activations halved</span><span class="q q-other"><b>Sacrifices</b>numerical accuracy</span><span class="q q-mem">Training state still $$16\Phi$$: FP32 parameters kept</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Activation checkpointing</b><span class="tag tag-trade">trade-off</span></div>
<p>Saves one layer's activations every $$\sqrt{L}$$ layers or so; when the backward pass needs the others, it reruns the forward pass from the nearest saved point.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: activations from $$L$$ layers down to about $$2\sqrt{L}$$ layers</span><span class="q q-calc"><b>Sacrifices</b>compute: one extra forward pass, $$R$$ drops to about 3/4</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">Mixed precision raises peak compute, but achieved compute is far below the peak, and the kernel implementation sets the gap</p>
<div class="lc-formula">\[ R = F \times u \]<p class="fnote">$$F$$: peak compute; $$u$$: utilization. Mixed precision raises $$F$$; the implementation of the kernels sets $$u$$.</p></div>
<ol class="lc-steps">
<li>The peak of BF16 Tensor Cores is about 15 times that of the FP32 units, but this is only an upper bound; achieved compute depends on each kernel.</li>
<li>The gap has two sources. The first is kernels limited by memory bandwidth: RMSNorm does about 1 FLOP per byte read or written, while H100 needs about 295 FLOP per byte (989 TFLOP/s ÷ 3.35 TB/s) to use its full compute; memory bandwidth sets its time, independent of peak compute.</li>
<li>These kernels (norm, softmax, activation functions, element-wise operations) do few operations but take much of the time: when training BERT-large with PyTorch on V100, they account for 0.2% of the operations and 39% of the time.</li>
<li>The second is matrix multiplies: the whole matrix does not fit on chip, and compute units wait for data to be read from GPU memory into the chip.</li>
<li>How to tell: a kernel that does $$W$$ FLOP and reads and writes $$Q$$ bytes of GPU memory takes at least the larger of $$W/F$$ and $$Q/B$$ ($$B$$ is the memory bandwidth; the derivation is in <a href="/notes/roofline/">the roofline post</a>). With arithmetic intensity $$W/Q$$ below the ridge point $$F/B$$, it is limited by memory bandwidth; above it, by compute.</li>
<li>The two kinds are optimized separately: for kernels limited by memory bandwidth, cut the bytes read and written to GPU memory (kernel fusion → online softmax → FlashAttention); for compute-limited matrix multiplies, cut the time compute units wait for data (tiling, pipelining).</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="kernels">Kernel efficiency</h2><span class="lc-sub">roofline · kernel fusion · FlashAttention · tiling and pipelining</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>For some kernels, memory bandwidth sets the time, independent of peak compute; matrix multiplies also wait for data to be read from GPU memory into the chip.</p></dd>
<dt>Solution</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>Kernel fusion</b><span class="tag tag-waste">waste removal</span></div>
<p>Merges several consecutive kernels limited by memory bandwidth into one kernel; intermediate results stay on chip and are not written back to GPU memory.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: bytes read and written to GPU memory</span><span class="q q-other"><b>Improves</b>fixed cost of kernel launches</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>FlashAttention</b><span class="tag tag-trade">trade-off</span></div>
<p>Attention computes a query, a key and a value vector for each token, stacks them into matrices $$Q$$, $$K$$ and $$V$$, then computes in turn $$S = QK^\top$$ (match scores between every pair of tokens), $$P = \text{softmax}(S)$$ and the output $$O = PV$$. $$S$$ and $$P$$ are both $$n \times n$$ matrices ($$n$$ is the sequence length), about 32 MB per head at $$n = 4096$$. Softmax needs the maximum and sum of a whole row; the standard implementation uses three kernels and writes $$S$$ and $$P$$ back to GPU memory.</p><p>FlashAttention uses online softmax to update each row's maximum and sum block by block, merges the three steps into one kernel and computes them in blocks on chip, so $$S$$ and $$P$$ are not written back to GPU memory; the backward pass recomputes $$S$$ and $$P$$ from $$Q$$, $$K$$ and these two per-row statistics. In the paper's example (GPT-2 medium, sequence length 1024, A100), the forward plus backward time of attention drops from 41.7 ms to 7.3 ms.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: GPU memory reads and writes 40.3 → 4.4 GB</span><span class="q q-mem"><b>Improves</b>memory: removes the $$n^2$$ term in activations</span><span class="q q-calc"><b>Sacrifices</b>compute: backward recomputes $$S$$ and $$P$$, 66.6 → 75.2 GFLOP</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Tiling and pipelining</b><span class="tag tag-waste">waste removal</span></div>
<p>In an $$n \times n$$ matrix multiply, each number takes part in $$n$$ multiply-adds; if each number (BF16) is read or written only once, the arithmetic intensity is $$n/3$$ (about 1365 at $$n = 4096$$). But the whole matrix does not fit on chip. With tiling, data read into the chip is reused many times; the next tile is read while the current one is computed.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: bytes read repeatedly from GPU memory</span><span class="q q-calc"><b>Improves</b>compute: time compute units wait for data</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">With peak and utilization both raised, one GPU's compute still has a limit, so the only option is more GPUs</p>
<p class="lc-num">6 × 405B × 15.6T tokens ≈ 3.8×10²⁵ FLOP; one H100 at 989 TFLOP/s would take about 1200 years</p>
<ol class="lc-steps">
<li>Both factors of $$R = F \times u$$ have been raised: mixed precision raises $$F$$ and kernel optimization raises $$u$$ (checkpointing goes the other way, trading about 1/3 extra compute for GPU memory).</li>
<li>The model and data set the numerator $$6\Phi D$$. Llama-3 405B: $$\Phi = 4.05 \times 10^{11}$$, $$D = 1.56 \times 10^{13}$$ tokens, $$6\Phi D \approx 3.8 \times 10^{25}$$ FLOP.</li>
<li>One H100 running continuously at its 989 TFLOP/s peak would need about 1200 years.</li>
<li>The numerator is fixed, $$R$$ cannot exceed the peak and $$\eta$$ cannot exceed 1, so only the number of GPUs $$N$$ can keep growing.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="ddp">DDP</h2><span class="lc-sub">Data Parallelism</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>One GPU's compute has a limit; frontier-scale training would take over a thousand years on one GPU.</p></dd>
<dt>Solution</dt><dd><p>Each of $$N$$ GPUs holds a full copy of the model and processes different data. Each step averages the gradients with all-reduce (each GPU contributes one piece of data, and at the end every GPU has their sum), and all copies apply the same update. Communication can overlap with the backward computation; when communication takes less time than computation, $$\eta$$ is close to 1.</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">trade-off</span><span class="q q-calc"><b>Improves</b>compute: each GPU computes $$1/N$$ of the data, training time about $$1/N$$</span><span class="q q-move"><b>Sacrifices</b>data movement: $$2\Phi$$ elements communicated per step</span><span class="q q-mem">Memory unchanged: each GPU still stores the full $$16\Phi$$, $$N$$ identical copies</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">DDP does not reduce GPU memory, and each GPU still has to hold the full training state. What if it does not fit?</p>
<p class="lc-num">7B × 16 bytes = 112 GB, more than the 80 GB of H100</p>
<ol class="lc-steps">
<li>DDP assumes that each GPU can hold the full training state of $$16\Phi$$ bytes.</li>
<li>$$16\Phi$$ grows linearly with the parameter count: for a 7B model it is 112 GB, more than the 80 GB of H100, and DDP cannot run.</li>
<li>Yet the $$N$$ copies of $$16\Phi$$ on the $$N$$ GPUs are identical.</li>
<li>So each GPU needs to store only $$1/N$$ and fetch the rest from other GPUs when needed, with no information lost.</li>
<li>ZeRO defines the order of sharding and the communication that fetching needs.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="zero">ZeRO / FSDP</h2><span class="lc-sub">Sharding the training state</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>$$16\Phi$$ exceeds one GPU's memory, while the $$N$$ GPUs store $$N$$ identical copies.</p></dd>
<dt>Solution</dt><dd><p>Shards in stages, from the least to the most frequently used data.</p><div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-1</b><span class="tag tag-waste">waste removal</span></div>
<p>Shards the optimizer states ($$12\Phi$$). All-reduce can be split into two steps: a reduce-scatter (each GPU gets $$1/N$$ of the gradient sum) and an all-gather (each GPU sends its $$1/N$$ to all GPUs). ZeRO-1 has each GPU update only its own $$1/N$$ of the parameters between the two steps and then all-gathers the updated parameters, so each GPU needs to store only $$1/N$$ of the optimizer states, and the communication volume is unchanged.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: $$16\Phi \to 4\Phi + 12\Phi/N$$</span><span class="q q-move">Data movement unchanged: $$2\Phi$$ communicated per step</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-2</b><span class="tag tag-waste">waste removal</span></div>
<p>Also shards the gradients ($$2\Phi$$): after the reduce-scatter, each GPU needs to keep only its own $$1/N$$ of the gradient sum and can free the rest.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: $$\to 2\Phi + 14\Phi/N$$</span><span class="q q-move">Data movement unchanged</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-3</b><span class="tag tag-trade">trade-off</span></div>
<p>Also shards the parameters ($$2\Phi$$): the full parameters of each layer are fetched before it is computed, once in the forward pass and once in the backward pass. The FULL_SHARD mode of PyTorch FSDP corresponds to ZeRO-3.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: $$\to 16\Phi/N$$</span><span class="q q-move"><b>Sacrifices</b>data movement: communication per step $$2\Phi \to 3\Phi$$</span></div>
</div>
</dd>
</dl>
<p class="lc-note">Memory is counted in bytes and communication in elements, following the convention of the ZeRO paper.</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">ZeRO splits storage, but each GPU still computes the full model, so with many GPUs, syncing parameters takes longer than computing</p>
<div class="lc-formula">\[ \frac{t_\text{comm}}{t_\text{comp}} = \frac{3\Phi \times 2\ \text{byte} \,/\, B_\text{net}}{6\Phi \cdot (G/N) \,/\, R} = \frac{R \cdot N}{B_\text{net} \cdot G} \]<p class="fnote">$$G$$: total tokens per step (global batch); $$B_\text{net}$$: cross-machine network bandwidth. $$\Phi$$ cancels out.</p></div>
<ol class="lc-steps">
<li>With ZeRO-3, each GPU communicates $$3\Phi$$ elements per step, which does not shrink with $$N$$; its computation is $$6\Phi$$ times the number of tokens it gets. The two can overlap; when the ratio in the formula above is below 1, communication can overlap completely with computation.</li>
<li>The total tokens per step $$G$$ has a limit: past the critical batch size, a larger batch saves fewer and fewer training steps.</li>
<li>With $$G$$ fixed, each GPU gets $$G/N$$ tokens, and the ratio of communication time to computation time is proportional to $$N$$.</li>
<li>Llama-3 405B uses 16384 H100s with $$G$$ = 16M tokens, only about 1000 tokens per GPU on average. If all 16384 GPUs used only ZeRO-3, at the measured rate of about 400 TFLOP/s per GPU, the ratio would be about 8.</li>
<li>Also, each GPU must process at least one full sequence: with long sequences, the activations of one sequence can exceed one GPU's memory capacity.</li>
<li>The first limit comes from each GPU computing the full model, and the second from each GPU processing full sequences. TP splits the matrix multiplies within a layer, so that several GPUs compute the same layer together; CP splits the sequence.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="tp-sp">TP · SP · CP</h2><span class="lc-sub">Intra-layer parallelism</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>With many GPUs, syncing parameters takes longer than computing; with long sequences, the activations of one sequence exceed one GPU's memory capacity.</p></dd>
<dt>Solution</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>TP</b><span class="tag tag-trade">trade-off</span></div>
<p>Tensor parallelism: splits each matrix multiply into $$t$$ parts across $$t$$ GPUs; the $$t$$ GPUs compute one copy of the model together, and $$N$$ in the formula above becomes the number of groups $$N/t$$.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: training state and the activations inside attention and FFN drop to $$1/t$$</span><span class="q q-move"><b>Improves</b>data movement: across machines, each GPU syncs only $$1/t$$ of the parameters</span><span class="q q-move"><b>Sacrifices</b>data movement: within a machine, 2 all-reduces per layer in each pass; the forward pass waits on them</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>SP</b><span class="tag tag-waste">waste removal</span></div>
<p>Sequence parallelism, here meaning the Megatron-LM approach: TP does not split LayerNorm and Dropout; each GPU stores its own copy of their activations, about 3/4 of a layer's activations at $$t = 8$$ (not counting the attention score matrices). SP splits them along the sequence into $$t$$ parts and rewrites the original all-reduce as a reduce-scatter and an all-gather of the same total volume.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: these activations drop to $$1/t$$</span><span class="q q-move">Data movement unchanged</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>CP</b><span class="tag tag-trade">trade-off</span></div>
<p>Context parallelism, the other class of sequence parallelism, for long sequences (Li et al. call it sequence parallelism; Ring Attention and DeepSpeed-Ulysses came later): the activations of the whole layer are split along the sequence into $$c$$ parts, and the K and V of other positions that attention needs are obtained through communication. Ring Attention passes them block by block around a ring, overlapped with computation; DeepSpeed-Ulysses switches to splitting by head with an all-to-all (each GPU sends different data to every other GPU); Llama-3 first all-gathers all K and V.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: activations of the whole layer drop to $$1/c$$</span><span class="q q-move"><b>Sacrifices</b>data movement: K and V sent in every layer</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">TP's communication limits its own scale, so crossing machines needs a split with little communication</p>
<p class="lc-num">NVLink within a machine 450 GB/s, cross-machine network 50 GB/s (one direction)</p>
<ol class="lc-steps">
<li>TP does 2 all-reduces per layer in each of the forward and backward passes; the forward pass must wait for each all-reduce to finish before it continues, so they are hard to overlap with computation.</li>
<li>This communication has to run on NVLink within a machine; the cross-machine network has only about 1/9 of its bandwidth. So TP is limited to one machine; a machine has 8 GPUs, so $$t \le 8$$.</li>
<li>With TP alone, 8 GPUs cannot hold a large model: the training state of 405B is about 6.5 TB, and 8 H100s have 640 GB in total. Combined with ZeRO-3, the number of groups is $$N/8$$, and at Llama-3's scale the ratio of communication time to computation time only drops from about 8 to about 1.</li>
<li>A split across machines needs little communication: PP splits by layer and passes only the activations at the boundaries between stages, so the volume is far smaller than the parameter count.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="pp">PP</h2><span class="lc-sub">Pipeline Parallelism</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>TP is limited to one machine; TP alone cannot hold a large model, and combined with ZeRO-3, syncing parameters across machines still takes about as long as computing.</p></dd>
<dt>Solution</dt><dd><p>Splits the $$L$$ layers into $$p$$ stages, which pass only activations between them. The batch is split into $$m$$ micro-batches fed in one after another, and the stages process different micro-batches at the same time; at the start and end, some GPUs wait, which is called the bubble, a fraction $$(p-1)/(m+p-1)$$ of the time. A larger $$m$$ shrinks the bubble, but with micro-batches that are too small, arithmetic intensity drops and fixed costs take a larger share.</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">trade-off</span><span class="q q-mem"><b>Improves</b>memory: each GPU stores only $$L/p$$ layers</span><span class="q q-move"><b>Improves</b>data movement: across machines, each GPU syncs only $$1/(t \cdot p)$$ of the parameters</span><span class="q q-calc"><b>Sacrifices</b>compute: bubble, about 19% at $$p = 16$$, $$m = 64$$</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">None of the three kinds of parallelism is enough alone, so they are combined</p>
<ol class="lc-steps">
<li>The limits of the three: TP is limited to one machine; PP has the bubble, and micro-batches cannot be too small; for DP (data parallelism, which covers both DDP and ZeRO), each doubling of the number of groups halves the tokens per group, while the global batch has a limit (the critical batch size).</li>
<li>The three limits have different sources: TP lacks cross-machine bandwidth, PP lacks enough micro-batches, and DP lacks room for the global batch to keep growing.</li>
<li>Because the sources differ, each kind of parallelism can be placed where its limit does not apply.</li>
<li>This largely fixes the combination: TP within a machine, PP across machines, DP at the outermost level. This is 3D parallelism.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="parallel-3d">3D parallelism</h2><span class="lc-sub">TP × PP × DP</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>TP is limited to one machine, PP has the bubble, and DP is limited by the global batch; none is enough alone.</p></dd>
<dt>Solution</dt><dd><p>$$N = t \times p \times d$$ ($$d$$ is the number of DP groups). One configuration of Llama-3 405B pretraining (sequence length 8K) uses $$8 \times 16 \times 128 = 16384$$ H100s, with FSDP for DP; the 128K-sequence stage changes to TP 8, CP 16, PP 16 and DP 8, which the report calls 4D parallelism. By the formula above, communication time ÷ computation time $$= R \cdot d/(B_\text{net} \cdot G)$$:</p></dd>
</dl>
<table class="lc-table">
<thead><tr><th>Split</th><th>Groups $$d$$</th><th>Communication time ÷ computation time</th></tr></thead>
<tbody>
<tr><td>ZeRO-3 only</td><td>16384</td><td><span class="q q-move">about 8</span></td></tr>
<tr><td>Add TP ($$t = 8$$)</td><td>2048</td><td><span class="q q-move">about 1</span></td></tr>
<tr><td>Then add PP ($$p = 16$$)</td><td>128</td><td><span class="q q-move">about 0.06</span></td></tr>
</tbody></table>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: cross-machine communication from about 8 times the computation time down to about 6%</span></div>
<p class="lc-note">Llama-3's FSDP does not free parameters after the forward pass and communicates $$2\Phi$$ elements per step; gradients are sent in FP32, for $$6\Phi$$ bytes in total, the same as the $$3\Phi \times 2$$ bytes used in the table.</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">As the parameter count keeps growing, can the operations per token stay the same?</p>
<ol class="lc-steps">
<li>3D parallelism solves the memory and training-time problems of large models.</li>
<li>Scaling laws give a reason to keep adding parameters: with the same data, more parameters give a trained model with lower loss (the gap between predictions and correct answers).</li>
<li>But in a dense model each token passes through all parameters: doubling the parameters doubles the operations per token, and doubles the cost of both training and inference.</li>
<li>The goal is to decouple the parameter count from the operations per token.</li>
<li>Most parameters are in the FFN (about 2/3 of each layer in the standard architecture), and each token passes through it separately: replacing the FFN with $$E$$ FFNs of the same structure and sending each token through only $$k$$ of them gives MoE.</li>
<li>With many experts, one GPU cannot hold all of them; with ZeRO-3, each layer would fetch the parameters of all $$E$$ experts, while each token uses only $$k$$ of them. Keeping the parameters in place and sending tokens to the GPUs that hold their experts is EP.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="moe">MoE · EP</h2><span class="lc-sub">Mixture of Experts · Expert Parallelism</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>In a dense model, the operations per token are proportional to the parameter count; with MoE, one GPU cannot hold all the experts, and with ZeRO-3 each layer has to fetch the parameters of all experts.</p></dd>
<dt>Solution</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>MoE</b><span class="tag tag-trade">trade-off</span></div>
<p>The FFN in each layer is replaced with $$E$$ FFNs of the same structure with separately trained parameters, called experts; a router picks $$k$$ of them for each token. The parameter count grows with $$E$$, and the operations per token grow only with $$k$$. DeepSeek-V3 has 671B parameters in total, and each token passes through 37B of them.</p>
<div class="lc-chips"><span class="q q-calc"><b>Improves</b>compute: $$\Phi$$ in $$6\Phi D$$ counts the parameters each token passes through, 671B → 37B</span><span class="q q-other"><b>Sacrifices</b>load balance: the router may concentrate tokens on a few experts, so extra balancing is needed</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>EP</b><span class="tag tag-trade">trade-off</span></div>
<p>Spreads the experts across GPUs, so each GPU stores only some of them; each token is sent to the GPU holding the expert it picked and sent back after the computation. Compared with ZeRO-3, the parameters stay in place and tokens are sent instead. In DeepSeek-V3, each MoE layer has 256 routed experts (experts chosen by the router), spread over 64 GPUs in training, 4 per GPU; in decode, each GPU holds only 1.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: each GPU stores only some of the experts</span><span class="q q-move"><b>Sacrifices</b>data movement: 2 all-to-alls per MoE layer in each of the forward and backward passes</span></div>
</div>
</dd>
</dl>
<p class="lc-note">The comparison here is with a dense model of the same parameter count, and the two use the same total GPU memory; compared with a dense model of the same operations per token, MoE stores about 634B more parameters, which EP spreads over more GPUs.</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">After deployment, the model generates tokens one by one for each request</p>
<p class="lc-num">Without storing intermediate results, the total operations to generate n tokens grow at least quadratically in n</p>
<ol class="lc-steps">
<li>Once trained, the model is deployed and generates answers to requests; inference has only forward computation.</li>
<li>Inference has two phases: prefill feeds the whole input into the model at once; decode generates new tokens one at a time, and computing the next token needs the previous one.</li>
<li>To generate the $$n$$-th token, attention needs the key and value vectors (K, V) of every earlier position.</li>
<li>Without storing them, each step recomputes the whole preceding context.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="kv-cache">KV cache</h2><span class="lc-sub">Storing K and V of earlier tokens</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>Generating the $$n$$-th token recomputes the previous $$n - 1$$ tokens.</p></dd>
<dt>Solution</dt><dd><p>Stores the K and V of each layer; each step computes only the new token, feeding in 1 token per request per step.</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">trade-off</span><span class="q q-calc"><b>Improves</b>compute: each step drops from the whole preceding context to 1 token</span><span class="q q-mem"><b>Sacrifices</b>memory: grows with length, 512 KiB per token for Llama 2 7B</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">After the KV cache, each decode step computes only 1 token but still reads all the weights once, and the bottleneck moves from compute to memory bandwidth</p>
<p class="lc-num">Llama 2 7B: 13.5 GB of weights read per step; at batch 1, under 1% of compute is used</p>
<div class="lc-formula">\[ T_\text{step} \approx \max\!\left(\frac{Q_\text{w} + b \cdot n \cdot q_\text{KV}}{B},\ \frac{2\Phi \cdot b}{F}\right) \]<p class="fnote">$$Q_\text{w}$$: bytes of the weights; $$b$$: number of requests computed together (batch); $$n$$: context length; $$q_\text{KV}$$: KV cache bytes per token.</p></div>
<ol class="lc-steps">
<li>In training, many tokens pass through the model together; the matrix multiplies are large and limited by compute.</li>
<li>Each decode step reads all the BF16 weights once and does 2 FLOP for every 2 bytes read; with $$b$$ requests computed together, the arithmetic intensity of the weight part is about $$b$$ FLOP/byte; when $$b$$ is far below the ridge point of 295, decode is limited by memory bandwidth.</li>
<li>At batch 1, under 1% of compute is used (details in <a href="/notes/roofline/">section 6 of the roofline post</a>). With a different bottleneck, training-side optimizations cannot be reused directly.</li>
<li>Following the formula above, the optimizations fall into two groups: increase $$b$$ so that each read of the weights generates more tokens; make each step shorter and the steps fewer.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="batching">Larger batches</h2><span class="lc-sub">Group 1</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>Each decode step reads all the weights once, only to generate 1 token for each of $$b$$ requests.</p></dd>
<dt>Limit</dt><dd><p>$$b$$ is limited by memory capacity: $$Q_\text{w} + b \cdot n \cdot q_\text{KV} \le$$ memory capacity. Llama 2 7B, a 4096-token context, an 80 GB H100: weights 13.5 GB, KV cache 2.1 GB per request, $$b$$ at most 30. Each step then reads about 78 GB, 64 GB of it KV cache; each request reads its own KV cache, so a larger $$b$$ does not amortize these reads.</p><p>GQA (several query heads share one set of K and V) and MLA (K and V compressed into short vectors) both reduce $$q_\text{KV}$$.</p></dd>
<dt>Solution</dt><dd><p>Of the four techniques, PagedAttention raises the $$b$$ that continuous batching can reach; chunked prefill and prefix caching are each independent.</p><div class="lc-tech"><div class="lc-tech-head"><b>Continuous batching</b><span class="tag tag-waste">waste removal</span></div>
<p>Requests vary in length, and starting and ending a whole batch together leaves empty slots. Instead, at every step, finished requests leave and new requests join.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: more useful tokens per weight read</span><span class="q q-other"><b>Improves</b>queueing time of new requests</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>PagedAttention</b><span class="tag tag-waste">waste removal</span></div>
<p>The limit of $$b$$ also depends on how well the KV cache memory is used. When memory is reserved for the maximum length, about 20% holds useful data (only about 38% even when the output length is known in advance). Instead, memory is allocated in fixed-size blocks, and the blocks need not be contiguous. Addressing by block makes the attention kernel about 20% to 26% slower, but $$b$$ can grow and end-to-end throughput rises 2 to 4 times, so this is counted as an implementation overhead.</p>
<div class="lc-chips"><span class="q q-mem"><b>Improves</b>memory: about 96% useful data</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Chunked prefill</b><span class="tag tag-trade">trade-off</span></div>
<p>When a new request's prefill is inserted, requests in decode wait for it to finish. Chunked prefill splits the prefill into chunks and computes one chunk together with decode each step, using the compute that decode leaves spare.</p>
<div class="lc-chips"><span class="q q-other"><b>Improves</b>decode stalls</span><span class="q q-other"><b>Sacrifices</b>latency of a new request's first token</span><span class="q q-move"><b>Sacrifices</b>data movement: each chunk rereads the KV cache of the chunks before it</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Prefix caching</b><span class="tag tag-waste">waste removal</span></div>
<p>Many requests share the same prefix (for example, a system prompt). Prefix caching keeps the KV cache of computed prefixes in GPU memory for reuse; the cache uses only spare GPU memory, and when the batch needs to grow, the least recently used parts are evicted first.</p>
<div class="lc-chips"><span class="q q-calc"><b>Improves</b>compute: repeated prefill</span><span class="q q-mem"><b>Improves</b>memory: one copy of each shared prefix</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">Independent of larger batches, Group 2 makes each step shorter and the steps fewer</p>
<div class="lc-formula"><p class="lc-wformula">Generation time of one request ≈ steps × bytes read per step ÷ achieved memory bandwidth</p></div>
<ol class="lc-steps">
<li>Larger batches let each read of the weights serve more requests; Group 2 shortens the generation time of a single request.</li>
<li>Each of the three factors above has one technique: Flash-Decoding raises the achieved bandwidth, quantization reduces the bytes read per step, and speculative decoding reduces the steps of the original model. The three can be used together.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="per-step">Shorter and fewer steps</h2><span class="lc-sub">Group 2</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>Larger batches raise throughput but do not shorten the generation time of a single request: that time is set by the number of steps, the bytes read per step and the achieved memory bandwidth.</p></dd>
<dt>Solution</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>Flash-Decoding</b><span class="tag tag-waste">waste removal</span></div>
<p>H100 has 132 SMs (groups of compute units that run independently); the more SMs read GPU memory at once, the closer the achieved bandwidth gets to the peak. FlashAttention divides work among SMs by batch, head and query block; in decode the query has only 1 position, and with a small batch there are fewer pieces of work than SMs. Flash-Decoding also splits the KV cache into chunks along its length, spreads them over more SMs, and at the end merges the chunks' results with a small kernel. The Flash-Decoding blog post reports that on A100 with long contexts, attention is up to about 50 times faster than with FlashAttention.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: idle SMs also read the KV cache, raising achieved bandwidth</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Quantization</b><span class="tag tag-trade">trade-off</span></div>
<p>Quantizes weights from BF16 to a lower precision such as INT4; when only weights are quantized, they are read in low precision and converted back to BF16 before the computation.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: weight bytes read per step about 1/4</span><span class="q q-calc"><b>Sacrifices</b>compute: converting back to BF16, using spare compute</span><span class="q q-other"><b>Sacrifices</b>numerical accuracy: GPTQ and similar methods use calibration data to reduce the loss of accuracy</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Speculative decoding</b><span class="tag tag-trade">trade-off</span></div>
<p>Generation yields 1 token per step, while verifying several given tokens takes only one forward pass. A small model first generates $$k$$ candidates, the original model verifies $$k+1$$ positions in one forward pass, and a rejection-sampling rule decides how many are accepted; the output distribution is the same as generating one by one. The gain depends on the fraction of candidates accepted; with a large batch, compute is no longer spare and the gain shrinks.</p>
<div class="lc-chips"><span class="q q-move"><b>Improves</b>data movement: the original model reads its weights less often (the small model also reads its weights, but they are far smaller)</span><span class="q q-calc"><b>Sacrifices</b>compute: verifying $$k+1$$ positions</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">Prefill and decode have different bottlenecks and interfere with each other on the same GPUs</p>
<ol class="lc-steps">
<li>Prefill processes the whole input at once and is limited by compute; decode is limited by memory bandwidth.</li>
<li>Chunked prefill computes the two together on the same GPUs and reduces decode stalls; but each decode step still waits for the prefill chunk in the same batch to finish, and the two must use the same parallelism and batch.</li>
<li>As with 3D parallelism, the two are placed separately according to the source of each limit.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="pd">P/D disaggregation</h2><span class="lc-sub">prefill and decode deployed separately</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>Prefill is limited by compute and decode by memory bandwidth; sharing the same GPUs, they interfere with each other.</p></dd>
<dt>Solution</dt><dd><p>Prefill and decode run on two sets of GPUs; after prefill, the KV cache is sent to the decode set, once per request. DistServe reports that, with 90% of requests meeting latency targets, the requests served per second per GPU are up to 7.4 times those of the compared systems, or latency targets 12.6 times tighter can be met.</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">trade-off</span><span class="q q-other"><b>Improves</b>latency: no interference, and separate parallelism and batch per phase</span><span class="q q-move"><b>Sacrifices</b>data movement: transferring the KV cache</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">Finally, RL in post-training puts the training and inference systems into one loop</p>
<ol class="lc-steps">
<li>At this point, training and inference each have their own system, each targeting its own bottleneck.</li>
<li>One RL step: the model generates answers (rollout), the answers are scored, and the scored samples update the parameters.</li>
<li>Generation is an inference workload, mostly decode, limited by memory bandwidth; the update is a training workload, limited by compute. One loop contains both workloads.</li>
<li>The training engine lacks inference optimizations such as KV cache management and continuous batching, so generating with it is slow.</li>
<li>The two systems share one set of parameters that is updated every step, so they must sync every step.</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="rl">RL post-training</h2><span class="lc-sub">Training engine and inference engine</span></div>
<dl class="lc-fact">
<dt>Problem</dt><dd><p>The training engine lacks inference optimizations such as KV cache management and continuous batching, so generation is slow.</p></dd>
<dt>Solution</dt><dd><p>Rollout uses an inference engine (vLLM, SGLang) and the update a training engine (FSDP, Megatron-LM); each step, the new parameters are synced to the inference engine and resharded to its partitioning. When run synchronously, $$T_\text{RL} = T_\text{rollout} + T_\text{update} + T_\text{sync}$$.</p><p>Two deployments: <b>colocated</b>, the two engines take turns on the same GPUs; <b>disaggregated</b>, the two engines are on different GPUs. When disaggregated, the next batch of rollouts can also start early with the previous version of the parameters; the parameters that generate the data are then one or more steps behind (off-policy), and the training algorithm must tolerate this difference.</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">trade-off</span><span class="q q-calc"><b>Improves</b>compute: the KV cache avoids recomputation during generation</span><span class="q q-move"><b>Improves</b>data movement: larger batches, more tokens per read of the weights</span><span class="q q-move"><b>Sacrifices</b>data movement: syncing parameters, 140 GB per step for a 70B model, up to 36.4% of a step in disaggregated deployment</span><span class="q q-other"><b>Sacrifices</b>simplicity: two engines to schedule</span></div>
</div>
</div>
</div>

## Summary {#recap}

<table class="lc-recap" markdown="0">
<thead><tr><th>Technique</th><th>Cause</th><th>Improves</th><th>Sacrifices</th><th>Type</th></tr></thead>
<tbody>
<tr class="lc-stage"><td colspan="5">Single-GPU training</td></tr>
<tr><td>Mixed precision</td><td>FP32 matrix multiplies skip Tensor Cores; activations take 4 bytes per number</td><td><span class="q q-calc">Compute</span><span class="q q-mem">Memory: activations</span></td><td><span class="q q-other">Numerical accuracy</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Checkpointing</td><td>Activations exceed memory capacity</td><td><span class="q q-mem">Memory: activations</span></td><td><span class="q q-calc">Compute: one extra forward pass</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Kernel fusion</td><td>Intermediate results read from and written to GPU memory repeatedly</td><td><span class="q q-move">Data movement</span><span class="q q-other">Kernel launches</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>FlashAttention</td><td>n × n S and P written back to GPU memory</td><td><span class="q q-move">Data movement</span><span class="q q-mem">Memory</span></td><td><span class="q q-calc">Compute: backward recomputation</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Tiling and pipelining</td><td>Matrix multiplies wait for data</td><td><span class="q q-move">Data movement</span><span class="q q-calc">Compute</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr class="lc-stage"><td colspan="5">Multi-GPU distributed training</td></tr>
<tr><td>DDP</td><td>One GPU's compute has a limit</td><td><span class="q q-calc">Compute: time about 1/N</span></td><td><span class="q q-move">Data movement: gradient sync</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>ZeRO-1/2</td><td>N GPUs store N identical copies of the training state</td><td><span class="q q-mem">Memory</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>ZeRO-3</td><td>Parameters still not sharded</td><td><span class="q q-mem">Memory</span></td><td><span class="q q-move">Data movement: 2Φ → 3Φ</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>TP</td><td>With many GPUs, syncing parameters takes longer than computing</td><td><span class="q q-mem">Memory</span><span class="q q-move">Data movement: cross-machine</span></td><td><span class="q q-move">Data movement: within a machine</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>SP (Megatron-LM)</td><td>Activations TP does not split are stored on every GPU</td><td><span class="q q-mem">Memory</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>CP</td><td>Long sequences: activations of one sequence do not fit</td><td><span class="q q-mem">Memory</span></td><td><span class="q q-move">Data movement: K and V</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>PP</td><td>TP is limited to one machine</td><td><span class="q q-mem">Memory</span><span class="q q-move">Data movement: cross-machine</span></td><td><span class="q q-calc">Compute: bubble</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>3D parallelism</td><td>Each of the three kinds of parallelism has a limit</td><td><span class="q q-move">Data movement: cross-machine</span></td><td>—</td><td>combination</td></tr>
<tr><td>MoE</td><td>Operations per token proportional to parameter count</td><td><span class="q q-calc">Compute</span></td><td><span class="q q-other">Load balance</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>EP</td><td>One GPU cannot hold all the experts</td><td><span class="q q-mem">Memory</span></td><td><span class="q q-move">Data movement: all-to-all</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr class="lc-stage"><td colspan="5">Inference</td></tr>
<tr><td>KV cache</td><td>Each step recomputes the whole preceding context</td><td><span class="q q-calc">Compute</span></td><td><span class="q q-mem">Memory</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Continuous batching</td><td>Empty slots in the batch</td><td><span class="q q-move">Data movement</span><span class="q q-other">Queueing time</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>PagedAttention</td><td>GPU memory reserved for the maximum length</td><td><span class="q q-mem">Memory</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>Chunked prefill</td><td>New requests' prefill stalls decode</td><td><span class="q q-other">Decode stalls</span></td><td><span class="q q-other">First-token latency</span><span class="q q-move">Data movement: rereading the KV cache</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Prefix caching</td><td>Repeated prefill of the same prefix</td><td><span class="q q-calc">Compute</span><span class="q q-mem">Memory</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>Flash-Decoding</td><td>With a small batch, few SMs read the KV cache</td><td><span class="q q-move">Data movement</span></td><td>—</td><td><span class="tag tag-waste">waste removal</span></td></tr>
<tr><td>Quantization</td><td>Each step reads all the weights once</td><td><span class="q q-move">Data movement</span></td><td><span class="q q-calc">Compute: converting back to BF16</span><span class="q q-other">Numerical accuracy</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>Speculative decoding</td><td>Each step yields only 1 token</td><td><span class="q q-move">Data movement: fewer steps</span></td><td><span class="q q-calc">Compute: verification</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr><td>P/D disaggregation</td><td>Prefill and decode share GPUs and interfere</td><td><span class="q q-other">Latency</span></td><td><span class="q q-move">Data movement: transferring the KV cache</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
<tr class="lc-stage"><td colspan="5">RL</td></tr>
<tr><td>Training engine + inference engine</td><td>The training engine generates slowly</td><td><span class="q q-calc">Compute</span><span class="q q-move">Data movement</span></td><td><span class="q q-move">Data movement: syncing parameters</span><span class="q q-other">Simplicity</span></td><td><span class="tag tag-trade">trade-off</span></td></tr>
</tbody></table>

## Sources {#sources}

<ol class="lc-refs" markdown="0">
<li><span class="topic">H100 compute, memory capacity and bandwidth, NVLink</span>: NVIDIA, <a href="https://www.nvidia.com/en-us/data-center/h100/">H100 Tensor Core GPU specifications (SXM)</a></li>
<li><span class="topic">H100 SM count (132)</span>: NVIDIA, <a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA Hopper Architecture In-Depth</a>, 2022</li>
<li><span class="topic">Mixed precision</span>: Micikevicius et al., <a href="https://arxiv.org/abs/1710.03740">Mixed Precision Training</a>, ICLR 2018</li>
<li><span class="topic">BF16 training</span>: Kalamkar et al., <a href="https://arxiv.org/abs/1905.12322">A Study of BFLOAT16 for Deep Learning Training</a>, 2019</li>
<li><span class="topic">activation checkpointing</span>: Chen et al., <a href="https://arxiv.org/abs/1604.06174">Training Deep Nets with Sublinear Memory Cost</a>, 2016</li>
<li><span class="topic">Activation estimate, Megatron-LM's SP</span>: Korthikanti et al., <a href="https://arxiv.org/abs/2205.05198">Reducing Activation Recomputation in Large Transformer Models</a>, MLSys 2023</li>
<li><span class="topic">GPT-3 XL architecture</span>: Brown et al., <a href="https://arxiv.org/abs/2005.14165">Language Models are Few-Shot Learners</a>, NeurIPS 2020</li>
<li><span class="topic">Shares of operations and time by kernel type (BERT-large, V100)</span>: Ivanov et al., <a href="https://arxiv.org/abs/2007.00072">Data Movement Is All You Need: A Case Study on Optimizing Transformers</a>, MLSys 2021</li>
<li><span class="topic">online softmax</span>: Milakov and Gimelshein, <a href="https://arxiv.org/abs/1805.02867">Online Normalizer Calculation for Softmax</a>, 2018</li>
<li><span class="topic">FlashAttention</span>: Dao et al., <a href="https://arxiv.org/abs/2205.14135">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</a>, NeurIPS 2022</li>
<li><span class="topic">Llama-3 405B parameters, token count, global batch, GPU count, network and parallel configuration</span>: Llama Team, <a href="https://arxiv.org/abs/2407.21783">The Llama 3 Herd of Models</a>, 2024</li>
<li><span class="topic">ZeRO</span>: Rajbhandari et al., <a href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a>, SC 2020</li>
<li><span class="topic">TP</span>: Shoeybi et al., <a href="https://arxiv.org/abs/1909.08053">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a>, 2019</li>
<li><span class="topic">CP (Li et al. call it sequence parallelism)</span>: Li et al., <a href="https://arxiv.org/abs/2105.13120">Sequence Parallelism: Long Sequence Training from System Perspective</a>, 2021</li>
<li><span class="topic">Ring Attention</span>: Liu et al., <a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>, 2023</li>
<li><span class="topic">DeepSpeed-Ulysses</span>: Jacobs et al., <a href="https://arxiv.org/abs/2309.14509">DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models</a>, 2023</li>
<li><span class="topic">Combining TP, PP and DP, and communication analysis</span>: Narayanan et al., <a href="https://arxiv.org/abs/2104.04473">Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</a>, SC 2021</li>
<li><span class="topic">PP and the bubble</span>: Huang et al., <a href="https://arxiv.org/abs/1811.06965">GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism</a>, NeurIPS 2019</li>
<li><span class="topic">critical batch size</span>: McCandlish et al., <a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>, 2018</li>
<li><span class="topic">scaling law</span>: Kaplan et al., <a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a>, 2020</li>
<li><span class="topic">MoE</span>: Shazeer et al., <a href="https://arxiv.org/abs/1701.06538">Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer</a>, ICLR 2017</li>
<li><span class="topic">EP</span>: Lepikhin et al., <a href="https://arxiv.org/abs/2006.16668">GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding</a>, ICLR 2021</li>
<li><span class="topic">DeepSeek-V3 parameter count and EP configuration</span>: DeepSeek-AI, <a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report</a>, 2024</li>
<li><span class="topic">Llama 2 7B layer count and width (32 layers, 4096)</span>: Touvron et al., <a href="https://arxiv.org/abs/2302.13971">LLaMA: Open and Efficient Foundation Language Models</a>, 2023</li>
<li><span class="topic">Llama 2 7B does not use GQA</span>: Touvron et al., <a href="https://arxiv.org/abs/2307.09288">Llama 2: Open Foundation and Fine-Tuned Chat Models</a>, 2023</li>
<li><span class="topic">continuous batching</span>: Yu et al., <a href="https://www.usenix.org/conference/osdi22/presentation/yu">Orca: A Distributed Serving System for Transformer-Based Generative Models</a>, OSDI 2022</li>
<li><span class="topic">PagedAttention</span>: Kwon et al., <a href="https://arxiv.org/abs/2309.06180">Efficient Memory Management for Large Language Model Serving with PagedAttention</a>, SOSP 2023</li>
<li><span class="topic">GQA</span>: Ainslie et al., <a href="https://arxiv.org/abs/2305.13245">GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</a>, EMNLP 2023</li>
<li><span class="topic">MLA</span>: DeepSeek-AI, <a href="https://arxiv.org/abs/2405.04434">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</a>, 2024</li>
<li><span class="topic">chunked prefill</span>: Agrawal et al., <a href="https://arxiv.org/abs/2403.02310">Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve</a>, OSDI 2024</li>
<li><span class="topic">prefix caching</span>: Zheng et al., <a href="https://arxiv.org/abs/2312.07104">SGLang: Efficient Execution of Structured Language Model Programs</a>, NeurIPS 2024</li>
<li><span class="topic">Flash-Decoding</span>: Dao et al., <a href="https://pytorch.org/blog/flash-decoding/">Flash-Decoding for long-context inference (PyTorch blog)</a>, 2023</li>
<li><span class="topic">Quantization</span>: Frantar et al., <a href="https://arxiv.org/abs/2210.17323">GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers</a>, ICLR 2023</li>
<li><span class="topic">speculative decoding</span>: Leviathan et al., <a href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</a>, ICML 2023</li>
<li><span class="topic">P/D disaggregation</span>: Zhong et al., <a href="https://arxiv.org/abs/2401.09670">DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving</a>, OSDI 2024</li>
<li><span class="topic">The two engines of RL and their deployment</span>: Sheng et al., <a href="https://arxiv.org/abs/2409.19256">HybridFlow: A Flexible and Efficient RLHF Framework</a>, EuroSys 2025</li>
<li><span class="topic">Asynchronous RL with rollouts starting early</span>: Noukhovitch et al., <a href="https://arxiv.org/abs/2410.18252">Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models</a>, ICLR 2025</li>
</ol>
