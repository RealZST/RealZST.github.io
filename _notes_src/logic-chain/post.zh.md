# LLM Infra 的逻辑链

<p class="lc-lead">本文面向<b>关心 LLM 系统设计、希望建立整体认识</b>的读者。从 LLM 本身出发，沿一条因果链贯穿训练与推理系统的主要设计，重点在每项设计的动机。每个主题依次说明<b>遇到了什么问题</b>、<b>怎么解决</b>、<b>优化了什么、牺牲了什么</b>；只保留理解设计所需的原理和数字，略去实现细节。数字都来自公开的硬件规格和论文，出处统一列在文末。</p>

<div class="lc-map" markdown="0">
<div class="lc-row">
<div class="lc-card">
<div class="lc-head"><h2 id="llm">LLM</h2></div>
<dl class="lc-fact">
<dt>范围</dt><dd><p>LLM（大语言模型）的训练与推理系统，以及其中主要设计的由来。</p></dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">要理解 LLM，先看它的 basic architecture 是什么</p>
<ol class="lc-steps">
<li>本文讨论 LLM 的训练与推理系统。</li>
<li>系统的开销由模型架构决定：分析开销之前，要先知道模型的架构。</li>
<li>当前的 LLM 几乎都采用同一种架构：Transformer。</li>
<li>因此先分析 Transformer 的架构，以及训练时 GPU 上的数据。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="transformer">Basic Architecture：Transformer</h2></div>
<dl class="lc-fact">
<dt>架构</dt><dd><p>每层两个主要模块：<b>attention</b>（让每个 token 汇总前文各个 token 的信息；内部分成若干个独立计算的 head）与 <b>FFN</b>（两层全连接网络，对每个 token 单独计算），$$L$$ 层堆叠；前后再加分词（把文本切成 token，即词或词的一部分）、embedding 等模块。大部分参数在这 $$L$$ 层的矩阵里。</p></dd>
<dt>数据</dt><dd><p>训练时 GPU 上的数据有四类：<b>参数</b>（上面这些矩阵本身，个数记为 $$\Phi$$）、<b>梯度</b>与<b>优化器状态</b>（每个参数配一个梯度，Adam 另配两个统计量），三者合称训练状态，大小只由模型决定；<b>激活</b>（每个模块前向的中间结果，反向时要用），大小随 batch（一次处理的序列数）和序列长度变化。输入的数据量很小，可以并入激活一起看。</p></dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">这些数据只有被存放、被计算、被搬运三种状态，每种状态对应一种开销</p>
<ol class="lc-steps">
<li>训练时 GPU 上的数据有四类：参数、梯度、优化器状态、激活。</li>
<li>任何一份数据，任何时刻都处于三种状态之一：正被存放、正被计算、正被搬运。</li>
<li>三种状态占用三种不同的硬件资源：存放占显存容量，计算占算力，搬运占带宽。三种资源各有一个上限，通常称为显存墙、算力墙、带宽墙。</li>
<li>三堵墙的单位分别是 Bytes、FLOP/s、Bytes/s。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="costs">三种开销</h2><span class="lc-sub">显存墙 · 算力墙 · 带宽墙</span></div>
<div class="lc-costs">
<div class="lc-cost mem"><span class="en">Data Placement</span><span class="def"><b>显存开销</b>：数据放在哪、占多少显存</span><span class="wall">显存墙 · <code>Bytes</code></span><span class="hw">H100：80 GB</span></div>
<div class="lc-cost calc"><span class="en">Data Calculation</span><span class="def"><b>计算开销</b>：在数据上执行的浮点运算</span><span class="wall">算力墙 · <code>FLOP/s</code></span><span class="hw">H100 BF16：989 TFLOP/s</span></div>
<div class="lc-cost move"><span class="en">Data Movement</span><span class="def"><b>搬运开销</b>：机器之间、GPU 之间、HBM 与 SRAM 之间的数据搬运</span><span class="wall">带宽墙 · <code>Bytes/s</code></span><span class="hw">H100：HBM 3.35 TB/s，NVLink 450 GB/s，跨机器网络 50 GB/s</span></div>
</div>
<p class="lc-note">三种开销不同类：显存开销是容量，超过显存容量就无法运行；计算开销和搬运开销是时间，等于运算量 ÷ 实际达到的算力、字节数 ÷ 实际达到的带宽，运算单元空闲、带宽没有用满时，时间变长。HBM 即显存；SRAM 是芯片内容量小、速度快的存储。H100 按 SXM 规格：算力是不含稀疏的稠密值；NVLink 和跨机器网络按单个方向计，跨机器网络按每个 GPU 一块 400 Gb/s 网卡计。</p>
<div class="lc-thesis">
<p><b>此后每项技术优化了什么、牺牲了什么，都用这三种开销描述。</b></p>
<p><b>判断顺序：</b>看到一项技术，按顺序问两个问题。① 它是不是去冗，即只去掉原本不起作用的部分？② 如果不是，它优化了哪种开销、牺牲了哪种开销？</p>
<p><span class="tag tag-waste">去冗</span> 去掉原本不起作用的部分，例如重复存放的数据、运算单元的空闲时间：优化一种开销，不牺牲其他开销；实现上的少量额外开销不计。</p>
<p><span class="tag tag-trade">交换</span> 优化一种开销，牺牲另一种。通常被优化的一项是当前场景的瓶颈，被牺牲的一项在这个场景下有富余。</p>
<p><b>三种开销之外：</b>每次启动 kernel（GPU 上执行的一个函数）或发起通信，另有微秒量级、与数据量无关的固定开销；少数技术牺牲的是数值误差、负载不均衡、延迟或调度的复杂度。卡片中以灰色标出。</p>
<p class="lc-legend">颜色表示变化的开销：<span class="q q-mem">显存开销</span> <span class="q q-calc">计算开销</span> <span class="q q-move">搬运开销</span> <span class="q q-other">三种开销之外</span></p>
</div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">有了三种开销就能分析交换，先从一个 GPU 内部开始</p>
<div class="lc-formula">\[ T = \frac{6\Phi D}{N \cdot R \cdot \eta} \qquad M = 16\Phi + A \]<p class="fnote">$$T$$：训练时间；$$D$$：训练的 token 数；$$N$$：GPU 数；$$R$$：每个 GPU 实际达到的算力（FLOP/s），只计模型本身的运算，重算不计入；$$\eta$$：并行效率。$$M$$：每个 GPU 的显存占用（byte），$$16\Phi$$ 是训练状态（参数、梯度和 Adam 的两个统计量，全用 FP32 时各 $$4\Phi$$ 字节），$$A$$ 是激活。</p></div>
<ol class="lc-steps">
<li>训练的总运算量约为 $$6\Phi D$$ FLOP：每个参数对每个 token，前向约 2 FLOP，反向约 4 FLOP。</li>
<li>分子由模型和数据决定，缩短训练时间只能增大分母的三个因子：单个 GPU 的实际算力 $$R$$（混合精度、算子优化），GPU 数 $$N$$（DDP 及之后的并行方法），并行效率 $$\eta$$（通信等待、GPU 空闲使它小于 1）。约束是每个 GPU 的显存占用 $$M$$ 不超过显存容量。</li>
<li>先改一个 GPU 上的两个默认设定，两者都是交换：用有富余的一种开销换瓶颈的一种，目的是提高 $$R$$ 或降低 $$M$$。</li>
<li>第一个默认设定是数值精度：每个数默认用 FP32 存，占 4 字节，矩阵乘不经过 Tensor Core（不启用 TF32 时）。H100 上 FP32 运算单元是 67 TFLOP/s，BF16 Tensor Core 是 989 TFLOP/s。</li>
<li>第二个是激活的保存方式：默认全部保存。1.3B 的模型（GPT-3 XL 的架构）一步处理 32 条 2048 token 的序列，不计 attention 的分数矩阵，激活按 BF16 计约 110 GB（计入约 500 GB），而训练状态只有 21 GB。</li>
<li>改变这两个设定的技术，分别是混合精度与 activation checkpointing。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="single-gpu">单 GPU 内的交换</h2><span class="lc-sub">混合精度 · activation checkpointing</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>全部用 FP32 时，矩阵乘的峰值只有 BF16 Tensor Core 的约 1/15，激活每个数占 4 字节；激活全部保存时，可能超过显存容量。</p></dd>
<dt>方案</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>混合精度</b><span class="tag tag-trade">交换</span></div>
<p>矩阵乘的输入和激活用 BF16（每个数 2 字节），乘积在 FP32 中累加；softmax、归一化等归约和参数更新用 FP32。参数更新留在 FP32，因为更新量小于参数的约 1/256 时，在 BF16 中会被舍掉。训练状态是 BF16 的参数和梯度各 $$2\Phi$$，加 FP32 的参数和 Adam 的两个统计量 $$12\Phi$$，合计 $$16\Phi$$ 字节。</p>
<div class="lc-chips"><span class="q q-calc"><b>优化</b>计算开销：矩阵乘峰值约 15 倍</span><span class="q q-mem"><b>优化</b>显存开销：激活减半</span><span class="q q-other"><b>牺牲</b>数值误差</span><span class="q q-mem">训练状态仍是 $$16\Phi$$：保留了 FP32 参数</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>Activation checkpointing</b><span class="tag tag-trade">交换</span></div>
<p>每隔约 $$\sqrt{L}$$ 层保存一层的激活；反向需要时，从最近的保存点重新做一遍前向。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：激活从 $$L$$ 层降到约 $$2\sqrt{L}$$ 层</span><span class="q q-calc"><b>牺牲</b>计算开销：多一次前向，$$R$$ 降到约 3/4</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">混合精度提高的是峰值算力，实际达到的却远低于峰值，差距由算子的实现决定</p>
<div class="lc-formula">\[ R = F \times u \]<p class="fnote">$$F$$：峰值算力；$$u$$：达成率。混合精度提高的是 $$F$$，$$u$$ 由算子的实现决定。</p></div>
<ol class="lc-steps">
<li>BF16 Tensor Core 的峰值是 FP32 运算单元的约 15 倍，但这只是上限；实际达到的算力取决于每个算子。</li>
<li>差距有两个来源。一是受显存带宽限制的算子：RMSNorm 每读写 1 字节约做 1 FLOP，而 H100 每字节要做约 295 FLOP（989 TFLOP/s ÷ 3.35 TB/s）才能用满算力；它的时间由显存带宽决定，与峰值算力无关。</li>
<li>这类算子（norm、softmax、激活函数、逐元素运算）运算量少，占用的时间却多：用 PyTorch 在 V100 上训练 BERT-large 时，它们占 0.2% 的运算量、39% 的时间。</li>
<li>二是矩阵乘：芯片内放不下整个矩阵，运算单元要等数据从显存读进芯片。</li>
<li>判断方法：一个算子做 $$W$$ FLOP、读写显存 $$Q$$ 字节，时间至少是 $$W/F$$ 与 $$Q/B$$ 中的较大者（$$B$$ 是显存带宽，推导见 <a href="/notes/roofline/zh/">roofline 一文</a>）。arithmetic intensity $$W/Q$$ 低于 ridge point $$F/B$$ 时受显存带宽限制，高于时受算力限制。</li>
<li>两类算子分开优化：受显存带宽限制的，减少读写显存的字节（算子融合 → online softmax → FlashAttention）；受算力限制的矩阵乘，减少运算单元等待数据的时间（分块、流水）。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="kernels">算子效率</h2><span class="lc-sub">roofline · 算子融合 · FlashAttention · 分块与流水</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>一部分算子的时间由显存带宽决定，与峰值算力无关；矩阵乘也要等数据从显存读进芯片。</p></dd>
<dt>方案</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>算子融合</b><span class="tag tag-waste">去冗</span></div>
<p>把连续几个受显存带宽限制的算子合成一个 kernel，中间结果留在芯片内，不写回显存。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：读写显存的字节</span><span class="q q-other"><b>优化</b>kernel 启动的固定开销</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>FlashAttention</b><span class="tag tag-trade">交换</span></div>
<p>attention 为每个 token 算出 query、key、value 三个向量，排成矩阵 $$Q$$、$$K$$、$$V$$，再依次算 $$S = QK^\top$$（token 两两之间的匹配分数）、$$P = \text{softmax}(S)$$、输出 $$O = PV$$。$$S$$、$$P$$ 都是 $$n \times n$$ 矩阵（$$n$$ 是序列长度），$$n = 4096$$ 时一个 head 约 32 MB。softmax 要用整行的最大值与总和，标准实现分三个 kernel，把 $$S$$、$$P$$ 写回显存。</p><p>FlashAttention 用 online softmax 逐块更新每行的最大值与总和，三步合进一个 kernel，在芯片内分块算完，$$S$$、$$P$$ 不写回显存；反向时用 $$Q$$、$$K$$ 和每行的这两个统计量重算 $$S$$、$$P$$。论文的例子（GPT-2 medium，序列长度 1024，A100）中，attention 的前向加反向时间从 41.7 ms 降到 7.3 ms。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：读写显存 40.3 → 4.4 GB</span><span class="q q-mem"><b>优化</b>显存开销：去掉激活中的 $$n^2$$ 项</span><span class="q q-calc"><b>牺牲</b>计算开销：反向重算 $$S$$、$$P$$，66.6 → 75.2 GFLOP</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>分块与流水</b><span class="tag tag-waste">去冗</span></div>
<p>$$n \times n$$ 的矩阵乘中每个数参与 $$n$$ 次乘加，每个数（BF16）只读写一次时 arithmetic intensity 是 $$n/3$$（$$n = 4096$$ 时约 1365）；但芯片内放不下整个矩阵。分块让读进芯片的数据多次使用；读下一块和算这一块同时进行。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：重复读显存的字节</span><span class="q q-calc"><b>优化</b>计算开销：运算单元等数据的时间</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">峰值和达成率都提高后，一个 GPU 的算力仍有上限，只能增加 GPU 数</p>
<p class="lc-num">6 × 405B × 15.6T token ≈ 3.8×10²⁵ FLOP；一个 H100 以 989 TFLOP/s 运行约 1200 年</p>
<ol class="lc-steps">
<li>$$R = F \times u$$ 的两个因子都已提高：混合精度提高 $$F$$，算子优化提高 $$u$$（checkpointing 方向相反，用约 1/3 的额外计算换显存）。</li>
<li>分子 $$6\Phi D$$ 由模型和数据决定。Llama-3 405B：$$\Phi = 4.05 \times 10^{11}$$，$$D = 1.56 \times 10^{13}$$ token，$$6\Phi D \approx 3.8 \times 10^{25}$$ FLOP。</li>
<li>一个 H100 以峰值 989 TFLOP/s 持续运行，需要约 1200 年。</li>
<li>分子固定，$$R$$ 不超过峰值，$$\eta$$ 不超过 1，能继续增大的只有 GPU 数 $$N$$。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="ddp">DDP</h2><span class="lc-sub">Data Parallelism</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>一个 GPU 的算力有上限，前沿规模的训练在一个 GPU 上需要上千年。</p></dd>
<dt>方案</dt><dd><p>$$N$$ 个 GPU 各放一份完整模型，处理不同的数据；每步用 all-reduce（每个 GPU 出一份数据，结束后都得到它们的和）求梯度平均，所有副本做相同的更新。通信可以和反向计算同时进行，通信时间短于计算时间时 $$\eta$$ 接近 1。</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">交换</span><span class="q q-calc"><b>优化</b>计算开销：每个 GPU 算 $$1/N$$ 的数据，训练时间约 $$1/N$$</span><span class="q q-move"><b>牺牲</b>搬运开销：每步通信 $$2\Phi$$ 个元素</span><span class="q q-mem">显存开销不变：每个 GPU 仍存完整的 $$16\Phi$$，$$N$$ 份完全相同</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">DDP 没有减少显存，每个 GPU 仍要放下完整的训练状态，放不下时怎么办？</p>
<p class="lc-num">7B × 16 byte = 112 GB，超过 H100 的 80 GB</p>
<ol class="lc-steps">
<li>DDP 的前提是每个 GPU 放得下完整的训练状态 $$16\Phi$$ 字节。</li>
<li>$$16\Phi$$ 随参数量线性增长：7B 模型是 112 GB，超过 H100 的 80 GB，DDP 无法运行。</li>
<li>而 $$N$$ 个 GPU 上的 $$N$$ 份 $$16\Phi$$ 完全相同。</li>
<li>因此每个 GPU 只需存 $$1/N$$，用到时从其他 GPU 取回，信息不丢失。</li>
<li>ZeRO 规定了分片的顺序，以及取回需要的通信。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="zero">ZeRO / FSDP</h2><span class="lc-sub">训练状态分片</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>$$16\Phi$$ 超出一个 GPU 的显存，而 $$N$$ 个 GPU 存的是 $$N$$ 份相同的副本。</p></dd>
<dt>方案</dt><dd><p>按使用频率从低到高逐级分片。</p><div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-1</b><span class="tag tag-waste">去冗</span></div>
<p>分片优化器状态（$$12\Phi$$）。all-reduce 可以拆成 reduce-scatter（每个 GPU 得到 $$1/N$$ 的梯度和）和 all-gather（每个 GPU 把自己的 $$1/N$$ 发给所有 GPU）两步；ZeRO-1 让每个 GPU 在两步之间只更新自己那 $$1/N$$ 参数，再 all-gather 更新后的参数，所以每个 GPU 只需存 $$1/N$$ 的优化器状态，通信量不变。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：$$16\Phi \to 4\Phi + 12\Phi/N$$</span><span class="q q-move">搬运开销不变：每步通信 $$2\Phi$$</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-2</b><span class="tag tag-waste">去冗</span></div>
<p>再分片梯度（$$2\Phi$$）：reduce-scatter 之后，每个 GPU 只需要保留自己那 $$1/N$$ 的梯度和，其余的可以释放。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：$$\to 2\Phi + 14\Phi/N$$</span><span class="q q-move">搬运开销不变</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>ZeRO-3</b><span class="tag tag-trade">交换</span></div>
<p>再分片参数（$$2\Phi$$）：每层计算前取回完整参数，前向、反向各一次。PyTorch FSDP 的 FULL_SHARD 模式对应 ZeRO-3。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：$$\to 16\Phi/N$$</span><span class="q q-move"><b>牺牲</b>搬运开销：每步通信 $$2\Phi \to 3\Phi$$</span></div>
</div>
</dd>
</dl>
<p class="lc-note">显存以字节计，通信量以元素个数计，沿用 ZeRO 论文的惯例。</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">ZeRO 切分的是存储，每个 GPU 仍要算完整的模型，所以 GPU 多时同步参数的时间超过计算时间</p>
<div class="lc-formula">\[ \frac{t_\text{comm}}{t_\text{comp}} = \frac{3\Phi \times 2\ \text{byte} \,/\, B_\text{net}}{6\Phi \cdot (G/N) \,/\, R} = \frac{R \cdot N}{B_\text{net} \cdot G} \]<p class="fnote">$$G$$：每步的总 token 数（global batch）；$$B_\text{net}$$：跨机器网络带宽。$$\Phi$$ 约掉了。</p></div>
<ol class="lc-steps">
<li>ZeRO-3 每个 GPU 每步通信 $$3\Phi$$ 个元素，不随 $$N$$ 减小；计算量是 $$6\Phi$$ 乘每个 GPU 分到的 token 数。两者可以重叠，上式的比值小于 1 时，通信可以完全与计算重叠。</li>
<li>每步的总 token 数 $$G$$ 有上限：超过 critical batch size 后，继续加大 batch，减少的训练步数越来越少。</li>
<li>$$G$$ 固定时，每个 GPU 分到 $$G/N$$ 个 token，通信与计算的时间比与 $$N$$ 成正比。</li>
<li>Llama-3 405B 用 16384 个 H100，$$G$$ = 16M token，平均每个 GPU 只有约 1000 个 token。若这 16384 个 GPU 全部只用 ZeRO-3，按实测的每个 GPU 约 400 TFLOP/s 计，比值约为 8。</li>
<li>另外，每个 GPU 至少要处理一条完整的序列：序列长时，一条序列的激活就可能超过一个 GPU 的显存容量。</li>
<li>第一个限制来自每个 GPU 计算完整的模型，第二个来自每个 GPU 处理完整的序列。TP 把层内的矩阵乘切开，改由几个 GPU 共同计算同一层；CP 把序列切开。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="tp-sp">TP · SP · CP</h2><span class="lc-sub">层内并行</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>GPU 多时，同步参数的时间超过计算时间；序列长时，一条序列的激活超过一个 GPU 的显存容量。</p></dd>
<dt>方案</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>TP</b><span class="tag tag-trade">交换</span></div>
<p>张量并行：把每个矩阵乘切成 $$t$$ 份，分给 $$t$$ 个 GPU；$$t$$ 个 GPU 共同算一份模型，上式的 $$N$$ 换成组数 $$N/t$$。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：训练状态和 attention、FFN 内部的激活降到 $$1/t$$</span><span class="q q-move"><b>优化</b>搬运开销：跨机器时每个 GPU 只同步 $$1/t$$ 的参数</span><span class="q q-move"><b>牺牲</b>搬运开销：机器内每层前向、反向各 2 次 all-reduce，前向要等它完成</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>SP</b><span class="tag tag-waste">去冗</span></div>
<p>序列并行，这里指 Megatron-LM 的做法：TP 不切分 LayerNorm 和 Dropout，这部分激活每个 GPU 各存一份，$$t = 8$$ 时约占一层激活的 3/4（不计 attention 的分数矩阵）。SP 把它沿序列切成 $$t$$ 份，原来的 all-reduce 改写成等量的 reduce-scatter 和 all-gather。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：这部分激活降到 $$1/t$$</span><span class="q q-move">搬运开销不变</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>CP</b><span class="tag tag-trade">交换</span></div>
<p>context parallelism，用于长序列的另一类序列并行（Li et al. 称为 sequence parallelism，此后有 Ring Attention、DeepSpeed-Ulysses）：整层的激活都沿序列切成 $$c$$ 份，attention 需要的其他位置的 K、V 通过通信获得。Ring Attention 沿环逐块传递，与计算重叠；DeepSpeed-Ulysses 用 all-to-all（每个 GPU 给其他每个 GPU 发不同的数据）改为按 head 切分；Llama-3 先 all-gather 全部 K、V。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：整层的激活降到 $$1/c$$</span><span class="q q-move"><b>牺牲</b>搬运开销：每层传 K、V</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">TP 的通信限制了它自身的规模，跨机器需要通信量小的切法</p>
<p class="lc-num">机器内 NVLink 450 GB/s，跨机器网络 50 GB/s（单向）</p>
<ol class="lc-steps">
<li>TP 每层前向、反向各 2 次 all-reduce；前向要等 all-reduce 完成才能继续计算，所以难以和计算重叠。</li>
<li>这种通信要放在机器内的 NVLink 上，跨机器网络的带宽只有它的约 1/9。因此 TP 限于一台机器内，一台机器 8 个 GPU，$$t \le 8$$。</li>
<li>单靠 TP，8 个 GPU 放不下大模型：405B 的训练状态约 6.5 TB，8 个 H100 共 640 GB。与 ZeRO-3 组合时组数是 $$N/8$$，Llama-3 的规模下通信与计算的时间比只从约 8 降到约 1。</li>
<li>跨机器的切法需要通信量小：PP 按层切开，段之间只传边界处的激活，传输量远小于模型的参数量。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="pp">PP</h2><span class="lc-sub">Pipeline Parallelism</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>TP 限于一台机器内；单靠 TP 放不下大模型，与 ZeRO-3 组合时，跨机器同步参数的时间仍约等于计算时间。</p></dd>
<dt>方案</dt><dd><p>把 $$L$$ 层分成 $$p$$ 段，段之间只传激活。batch 切成 $$m$$ 个 micro-batch 依次送入，各段同时处理不同的 micro-batch；开头和结尾部分 GPU 在等待，称为 bubble，占比 $$(p-1)/(m+p-1)$$。增大 $$m$$ 可以压低 bubble，但 micro-batch 太小时 arithmetic intensity 降低、固定开销占比增大。</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">交换</span><span class="q q-mem"><b>优化</b>显存开销：每个 GPU 只存 $$L/p$$ 层</span><span class="q q-move"><b>优化</b>搬运开销：跨机器时每个 GPU 只同步 $$1/(t \cdot p)$$ 的参数</span><span class="q q-calc"><b>牺牲</b>计算开销：bubble，$$p = 16$$、$$m = 64$$ 时约 19%</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">三种并行单独用都不够，只能组合</p>
<ol class="lc-steps">
<li>三种并行各自的限制：TP 限于一台机器内；PP 有 bubble，micro-batch 不能太小；DP（数据并行，DDP 与 ZeRO 都属于这一类）的组数每增加一倍，每组分到的 token 减半，而 global batch 有上限（critical batch size）。</li>
<li>三个限制的来源各不相同：TP 缺的是跨机器的带宽，PP 缺的是足够多的 micro-batch，DP 缺的是 global batch 继续增大的余地。</li>
<li>来源不同，所以每种并行可以放在它的限制不起作用的位置。</li>
<li>于是组合方式基本确定：TP 在机器内，PP 跨机器，DP 在最外层，即 3D 并行。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="parallel-3d">3D 并行</h2><span class="lc-sub">TP × PP × DP</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>TP 限于机器内，PP 有 bubble，DP 受 global batch 限制，单独使用都不够。</p></dd>
<dt>方案</dt><dd><p>$$N = t \times p \times d$$（$$d$$ 是 DP 的组数）。Llama-3 405B 预训练的一种配置（序列长度 8K）用 $$8 \times 16 \times 128 = 16384$$ 个 H100，DP 用 FSDP；序列长度 128K 的阶段改为 TP 8、CP 16、PP 16、DP 8，报告称为 4D 并行。按上面的式子，通信时间 ÷ 计算时间 $$= R \cdot d/(B_\text{net} \cdot G)$$：</p></dd>
</dl>
<table class="lc-table">
<thead><tr><th>切法</th><th>组数 $$d$$</th><th>通信时间 ÷ 计算时间</th></tr></thead>
<tbody>
<tr><td>全部 ZeRO-3</td><td>16384</td><td><span class="q q-move">约 8</span></td></tr>
<tr><td>加 TP（$$t = 8$$）</td><td>2048</td><td><span class="q q-move">约 1</span></td></tr>
<tr><td>再加 PP（$$p = 16$$）</td><td>128</td><td><span class="q q-move">约 0.06</span></td></tr>
</tbody></table>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：跨机器通信从计算时间的约 8 倍降到约 6%</span></div>
<p class="lc-note">Llama-3 的 FSDP 前向后不释放参数，每步通信 $$2\Phi$$ 个元素，梯度按 FP32 传，合计 $$6\Phi$$ 字节，与按 $$3\Phi \times 2$$ 字节计的结果相同。</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">参数量继续增加时，每个 token 的运算量能否保持不变？</p>
<ol class="lc-steps">
<li>3D 并行解决了大模型的显存与训练时间问题。</li>
<li>scaling law 给出继续增加参数的理由：同样数据下，参数越多，训练出的模型 loss（预测与正确答案的差距）越低。</li>
<li>但 dense 模型里每个 token 经过全部参数：参数翻倍，每个 token 的运算量也翻倍，训练和推理的成本都翻倍。</li>
<li>目标是让参数量与每个 token 的运算量解耦。</li>
<li>参数的大部分在 FFN（标准架构中约占每层的 2/3），而且每个 token 单独经过它：把 FFN 换成 $$E$$ 个结构相同的 FFN、每个 token 只经过其中 $$k$$ 个，就是 MoE。</li>
<li>expert 多了，一个 GPU 放不下全部 expert；若用 ZeRO-3，每层要取回全部 $$E$$ 个 expert 的参数，而每个 token 只用其中 $$k$$ 个。改为参数不动、把 token 发到 expert 所在的 GPU，就是 EP。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="moe">MoE · EP</h2><span class="lc-sub">Mixture of Experts · Expert Parallelism</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>dense 模型中，每个 token 的运算量与参数量成正比；改用 MoE 后，一个 GPU 放不下全部 expert，用 ZeRO-3 每层又要取回全部 expert 的参数。</p></dd>
<dt>方案</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>MoE</b><span class="tag tag-trade">交换</span></div>
<p>每层的 FFN 换成 $$E$$ 个结构相同、参数各自训练的 FFN，叫 expert；router 为每个 token 选其中 $$k$$ 个。参数量随 $$E$$ 增长，每个 token 的运算量只随 $$k$$ 增长。DeepSeek-V3 共 671B 参数，每个 token 经过其中 37B。</p>
<div class="lc-chips"><span class="q q-calc"><b>优化</b>计算开销：$$6\Phi D$$ 中的 $$\Phi$$ 按每个 token 经过的参数计，671B → 37B</span><span class="q q-other"><b>牺牲</b>负载不均衡：router 可能把 token 集中发给少数 expert，要额外均衡</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>EP</b><span class="tag tag-trade">交换</span></div>
<p>把 expert 分到不同的 GPU 上，每个 GPU 只存一部分 expert；token 发到它选中的 expert 所在的 GPU，算完再发回。与 ZeRO-3 相比，参数不动，传的是 token。DeepSeek-V3 每个 MoE 层有 256 个 routed expert（由 router 选择的 expert），训练时分到 64 个 GPU 上，每个 GPU 4 个；decode 时每个 GPU 只放 1 个 expert。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：每个 GPU 只存一部分 expert</span><span class="q q-move"><b>牺牲</b>搬运开销：每个 MoE 层前向、反向各 2 次 all-to-all</span></div>
</div>
</dd>
</dl>
<p class="lc-note">这里与参数量相同的 dense 模型比较，两者的总显存相同；若与每个 token 运算量相同的 dense 模型比较，MoE 要多存约 634B 参数，由 EP 分到更多 GPU 上。</p>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">模型部署后，按请求逐个生成 token</p>
<p class="lc-num">不存中间结果时，生成 n 个 token 的总运算量至少随 n² 增长</p>
<ol class="lc-steps">
<li>模型训练完成后部署上线，按请求生成回答；推理只有前向计算。</li>
<li>推理分两个阶段：prefill 把整段输入一次送进模型；decode 逐个生成新 token，下一个 token 的计算要用到上一个。</li>
<li>生成第 $$n$$ 个 token 时，attention 要用前面每个位置的 key、value 向量（K、V）。</li>
<li>不存下来，每步都要重算整段前文。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="kv-cache">KV cache</h2><span class="lc-sub">存下前文的 K、V</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>生成第 $$n$$ 个 token 时，要把前面 $$n - 1$$ 个 token 重算一遍。</p></dd>
<dt>方案</dt><dd><p>把每层的 K、V 存下来，每步只算新 token，每个请求每步送进 1 个 token。</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">交换</span><span class="q q-calc"><b>优化</b>计算开销：每步从整段前文降到 1 个 token</span><span class="q q-mem"><b>牺牲</b>显存开销：随长度增长，Llama 2 7B 每 token 512 KiB</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">KV cache 之后，decode 每步只算 1 个 token，却要读一遍全部权重，瓶颈从算力变为显存带宽</p>
<p class="lc-num">Llama 2 7B：每步读 13.5 GB 权重；batch 为 1 时，算力用到不足 1%</p>
<div class="lc-formula">\[ T_\text{step} \approx \max\!\left(\frac{Q_\text{w} + b \cdot n \cdot q_\text{KV}}{B},\ \frac{2\Phi \cdot b}{F}\right) \]<p class="fnote">$$Q_\text{w}$$：权重的字节数；$$b$$：一起算的请求数（batch）；$$n$$：上下文长度；$$q_\text{KV}$$：每 token 的 KV cache 字节数。</p></div>
<ol class="lc-steps">
<li>训练时大量 token 一起经过模型，矩阵乘规模大，受算力限制。</li>
<li>decode 每步读一遍全部 BF16 权重，每读 2 字节做 2 FLOP；$$b$$ 个请求一起算时，权重部分的 arithmetic intensity 约 $$b$$ FLOP/byte；$$b$$ 远小于 ridge point 295 时受显存带宽限制。</li>
<li>batch 为 1 时，算力用到不足 1%（展开见 <a href="/notes/roofline/zh/">roofline 一文第 6 节</a>）。瓶颈不同，训练侧的优化不能直接沿用。</li>
<li>按上式分两组优化：加大 $$b$$，使每次读权重生成更多 token；缩短每一步、减少步数。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="batching">加大 batch</h2><span class="lc-sub">第一组</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>decode 每步读一遍全部权重，只为 $$b$$ 个请求各生成 1 个 token。</p></dd>
<dt>上限</dt><dd><p>$$b$$ 受显存容量限制：$$Q_\text{w} + b \cdot n \cdot q_\text{KV} \le$$ 显存容量。Llama 2 7B、上下文 4096 token、80 GB 的 H100：权重 13.5 GB，每个请求 KV cache 2.1 GB，$$b$$ 最多 30。这时每步读约 78 GB，其中 64 GB 是 KV cache；每个请求各读自己的 KV cache，加大 $$b$$ 不能分摊这部分读取。</p><p>GQA（多个 query head 共用一组 K、V）和 MLA（把 K、V 压缩成短向量）都能减小 $$q_\text{KV}$$。</p></dd>
<dt>方案</dt><dd><p>四项技术中，PagedAttention 提高 continuous batching 能达到的 $$b$$；chunked prefill 与 prefix caching 各自独立。</p><div class="lc-tech"><div class="lc-tech-head"><b>continuous batching</b><span class="tag tag-waste">去冗</span></div>
<p>请求长短不一，整批一起开始、一起结束会留下空位。改为每一步都让完成的请求退出、新请求加入。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：读一遍权重生成更多有效 token</span><span class="q q-other"><b>优化</b>新请求的排队时间</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>PagedAttention</b><span class="tag tag-waste">去冗</span></div>
<p>$$b$$ 的上限还取决于 KV cache 显存的利用率。按最大长度预留时，约 20% 存放有效数据（预先知道输出长度也只有约 38%）。改为按固定大小的块分配，块不要求连续。按块寻址让 attention kernel 慢约 20% 到 26%，但 $$b$$ 可以加大，端到端吞吐提高 2 到 4 倍，所以计为实现开销。</p>
<div class="lc-chips"><span class="q q-mem"><b>优化</b>显存开销：有效数据约 96%</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>chunked prefill</b><span class="tag tag-trade">交换</span></div>
<p>新请求的 prefill 插入时，正在 decode 的请求要等它算完。把 prefill 切成块，每步和 decode 一起算一块，用的是 decode 时有余量的那部分算力。</p>
<div class="lc-chips"><span class="q q-other"><b>优化</b>decode 的停顿</span><span class="q q-other"><b>牺牲</b>新请求第一个 token 的延迟</span><span class="q q-move"><b>牺牲</b>搬运开销：每块都要重读前面各块的 KV cache</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>prefix caching</b><span class="tag tag-waste">去冗</span></div>
<p>很多请求的前缀相同（例如系统提示）。把算过的前缀的 KV cache 留在显存里复用；缓存只占空余的显存，需要加大 batch 时先淘汰最久未用的部分。</p>
<div class="lc-chips"><span class="q q-calc"><b>优化</b>计算开销：重复的 prefill</span><span class="q q-mem"><b>优化</b>显存开销：相同前缀只存一份</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">第二组与加大 batch 互不依赖，作用是缩短每一步、减少步数</p>
<div class="lc-formula"><p class="lc-wformula">一个请求的生成时间 ≈ 步数 × 每步读的字节 ÷ 实际达到的显存带宽</p></div>
<ol class="lc-steps">
<li>加大 batch 让每次读权重服务更多请求；第二组缩短单个请求的生成时间。</li>
<li>上式三个因子各有一项技术：Flash-Decoding 提高实际达到的带宽，量化减少每步读的字节，speculative decoding 减少原模型的步数。三项可以同时使用。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="per-step">缩短每步、减少步数</h2><span class="lc-sub">第二组</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>加大 batch 提高吞吐，但不缩短单个请求的生成时间：它由步数、每步读的字节和实际达到的显存带宽决定。</p></dd>
<dt>方案</dt><dd><div class="lc-tech"><div class="lc-tech-head"><b>Flash-Decoding</b><span class="tag tag-waste">去冗</span></div>
<p>H100 有 132 个 SM（独立执行的运算单元组），同时读显存的 SM 越多，实际带宽越接近峰值。FlashAttention 按 batch、head 和 query 块把工作分给 SM；decode 时 query 只有 1 个位置，batch 小时分出的工作少于 SM 数。Flash-Decoding 再把 KV cache 沿长度切块，分给更多 SM，最后用一个小 kernel 合并各块的结果；Flash-Decoding 的博客报告，在 A100 上、长上下文时，attention 比 FlashAttention 最多快约 50 倍。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：空闲的 SM 也参与读 KV cache，实际带宽提高</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>量化</b><span class="tag tag-trade">交换</span></div>
<p>权重从 BF16 量化为 INT4 等低精度；只量化权重时，读进来先转回 BF16 再算。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：每步读的权重字节约 1/4</span><span class="q q-calc"><b>牺牲</b>计算开销：转回 BF16，用的是有余量的算力</span><span class="q q-other"><b>牺牲</b>数值误差：GPTQ 等方法用校准数据减小</span></div>
</div>
<div class="lc-tech"><div class="lc-tech-head"><b>speculative decoding</b><span class="tag tag-trade">交换</span></div>
<p>生成每步得到 1 个 token，而验证多个给定的 token 只需一次前向。先用小模型生成 $$k$$ 个候选，原模型一次前向验证 $$k+1$$ 个位置，按拒绝采样规则决定接受到第几个，输出分布和逐个生成相同。收益取决于候选被接受的比例；batch 大时算力不再有余量，收益减小。</p>
<div class="lc-chips"><span class="q q-move"><b>优化</b>搬运开销：原模型读权重的次数减少（小模型也要读权重，但小得多）</span><span class="q q-calc"><b>牺牲</b>计算开销：验证 $$k+1$$ 个位置</span></div>
</div>
</dd>
</dl>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">prefill 和 decode 的瓶颈不同，放在同一组 GPU 上会互相影响</p>
<ol class="lc-steps">
<li>prefill 一次处理整段输入，受算力限制；decode 受显存带宽限制。</li>
<li>chunked prefill 让两者在同一组 GPU 上一起算，减少了 decode 的停顿；但每个 decode 步仍要等同批的 prefill 块算完，两者也只能用同一种并行方式和 batch。</li>
<li>和 3D 并行一样，按限制的来源分开放置。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="pd">P/D 分离</h2><span class="lc-sub">prefill 与 decode 分开部署</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>prefill 受算力限制，decode 受显存带宽限制，共用一组 GPU 时互相影响。</p></dd>
<dt>方案</dt><dd><p>prefill 和 decode 放在两组 GPU 上，prefill 算完把 KV cache 传给 decode 组，每个请求只传一次。DistServe 报告，在 90% 的请求满足延迟要求的条件下，每个 GPU 每秒可服务的请求数最多是对比系统的 7.4 倍，或者能满足严格 12.6 倍的延迟要求。</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">交换</span><span class="q q-other"><b>优化</b>延迟：两个阶段不再互相影响，各自选择并行方式和 batch</span><span class="q q-move"><b>牺牲</b>搬运开销：传输 KV cache</span></div>
</div>
</div>
<div class="lc-row">
<div class="lc-edge">
<p class="lc-q">最后，post-training 的 RL 把训练和推理两套系统放进同一个循环</p>
<ol class="lc-steps">
<li>至此，训练和推理各有一套系统，各自针对自己的瓶颈。</li>
<li>RL 的一步是：模型生成回答（rollout），给回答打分，用打过分的样本更新参数。</li>
<li>生成是推理负载，大部分时间在 decode，受显存带宽限制；更新是训练负载，受算力限制。一个循环同时包含两种负载。</li>
<li>训练引擎缺少 KV cache 管理、continuous batching 等推理优化，用它生成很慢。</li>
<li>两套系统共用一份每步都在更新的参数，需要每步同步。</li>
</ol>
</div>
<div class="lc-card">
<div class="lc-head"><h2 id="rl">RL post-training</h2><span class="lc-sub">训练引擎与推理引擎</span></div>
<dl class="lc-fact">
<dt>问题</dt><dd><p>训练引擎缺少 KV cache 管理、continuous batching 等推理优化，生成慢。</p></dd>
<dt>方案</dt><dd><p>rollout 用推理引擎（vLLM、SGLang），更新用训练引擎（FSDP、Megatron-LM），每步把新参数同步给推理引擎，并按推理引擎的切分方式重新切分。同步执行时，$$T_\text{RL} = T_\text{rollout} + T_\text{update} + T_\text{sync}$$。</p><p>两种部署：<b>共置</b>，两个引擎在同一组 GPU 上轮流运行；<b>分离</b>，两个引擎在不同的 GPU 上。分离时还可以让下一批 rollout 用上一版参数提前开始，这时生成数据的参数落后一步或更多（off-policy），训练算法要能容忍这种差异。</p></dd>
</dl>
<div class="lc-chips"><span class="tag tag-trade">交换</span><span class="q q-calc"><b>优化</b>计算开销：生成时 KV cache 免去重算</span><span class="q q-move"><b>优化</b>搬运开销：batch 更大，每次读权重生成更多 token</span><span class="q q-move"><b>牺牲</b>搬运开销：同步参数，70B 模型每步 140 GB，分离部署时最多占一步时间的 36.4%</span><span class="q q-other"><b>牺牲</b>调度两个引擎的复杂度</span></div>
</div>
</div>
</div>

## 汇总 {#recap}

<table class="lc-recap" markdown="0">
<thead><tr><th>技术</th><th>起因</th><th>优化</th><th>牺牲</th><th>类型</th></tr></thead>
<tbody>
<tr class="lc-stage"><td colspan="5">单 GPU 训练</td></tr>
<tr><td>混合精度</td><td>FP32 的矩阵乘不经过 Tensor Core，激活每个数占 4 字节</td><td><span class="q q-calc">计算</span><span class="q q-mem">显存：激活</span></td><td><span class="q q-other">数值误差</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>checkpointing</td><td>激活超过显存容量</td><td><span class="q q-mem">显存：激活</span></td><td><span class="q q-calc">计算：多一次前向</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>算子融合</td><td>中间结果反复读写显存</td><td><span class="q q-move">搬运</span><span class="q q-other">kernel 启动</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>FlashAttention</td><td>n × n 的 S、P 写回显存</td><td><span class="q q-move">搬运</span><span class="q q-mem">显存</span></td><td><span class="q q-calc">计算：反向重算</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>分块与流水</td><td>矩阵乘等待数据</td><td><span class="q q-move">搬运</span><span class="q q-calc">计算</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr class="lc-stage"><td colspan="5">多 GPU 分布式训练</td></tr>
<tr><td>DDP</td><td>一个 GPU 的算力有上限</td><td><span class="q q-calc">计算：时间约 1/N</span></td><td><span class="q q-move">搬运：同步梯度</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>ZeRO-1/2</td><td>N 个 GPU 存 N 份相同的训练状态</td><td><span class="q q-mem">显存</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>ZeRO-3</td><td>参数仍未分片</td><td><span class="q q-mem">显存</span></td><td><span class="q q-move">搬运：2Φ → 3Φ</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>TP</td><td>GPU 多时同步参数的时间超过计算</td><td><span class="q q-mem">显存</span><span class="q q-move">搬运：跨机器</span></td><td><span class="q q-move">搬运：机器内</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>SP（Megatron-LM）</td><td>TP 不切分的激活每个 GPU 各存一份</td><td><span class="q q-mem">显存</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>CP</td><td>序列长，一条序列的激活放不下</td><td><span class="q q-mem">显存</span></td><td><span class="q q-move">搬运：传 K、V</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>PP</td><td>TP 限于一台机器</td><td><span class="q q-mem">显存</span><span class="q q-move">搬运：跨机器</span></td><td><span class="q q-calc">计算：bubble</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>3D 并行</td><td>三种并行各有限制</td><td><span class="q q-move">搬运：跨机器</span></td><td>—</td><td>组合</td></tr>
<tr><td>MoE</td><td>每个 token 的运算量与参数量成正比</td><td><span class="q q-calc">计算</span></td><td><span class="q q-other">负载不均衡</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>EP</td><td>一个 GPU 放不下全部 expert</td><td><span class="q q-mem">显存</span></td><td><span class="q q-move">搬运：all-to-all</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr class="lc-stage"><td colspan="5">推理</td></tr>
<tr><td>KV cache</td><td>每步重算整段前文</td><td><span class="q q-calc">计算</span></td><td><span class="q q-mem">显存</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>continuous batching</td><td>批内留下空位</td><td><span class="q q-move">搬运</span><span class="q q-other">排队时间</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>PagedAttention</td><td>按最大长度预留显存</td><td><span class="q q-mem">显存</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>chunked prefill</td><td>新请求的 prefill 让 decode 停顿</td><td><span class="q q-other">decode 的停顿</span></td><td><span class="q q-other">首 token 延迟</span><span class="q q-move">搬运：重读 KV cache</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>prefix caching</td><td>相同的前缀重复 prefill</td><td><span class="q q-calc">计算</span><span class="q q-mem">显存</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>Flash-Decoding</td><td>batch 小时读 KV cache 的 SM 少</td><td><span class="q q-move">搬运</span></td><td>—</td><td><span class="tag tag-waste">去冗</span></td></tr>
<tr><td>量化</td><td>每步读一遍全部权重</td><td><span class="q q-move">搬运</span></td><td><span class="q q-calc">计算：转回 BF16</span><span class="q q-other">数值误差</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>speculative decoding</td><td>每步只得到 1 个 token</td><td><span class="q q-move">搬运：步数减少</span></td><td><span class="q q-calc">计算：验证</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr><td>P/D 分离</td><td>prefill 与 decode 共用 GPU，互相影响</td><td><span class="q q-other">延迟</span></td><td><span class="q q-move">搬运：传 KV cache</span></td><td><span class="tag tag-trade">交换</span></td></tr>
<tr class="lc-stage"><td colspan="5">RL</td></tr>
<tr><td>训练引擎 + 推理引擎</td><td>训练引擎生成慢</td><td><span class="q q-calc">计算</span><span class="q q-move">搬运</span></td><td><span class="q q-move">搬运：同步参数</span><span class="q q-other">调度复杂度</span></td><td><span class="tag tag-trade">交换</span></td></tr>
</tbody></table>

## 出处 {#sources}

<ol class="lc-refs" markdown="0">
<li><span class="topic">H100 的算力、显存容量与带宽、NVLink</span>：NVIDIA，<a href="https://www.nvidia.com/en-us/data-center/h100/">H100 Tensor Core GPU 产品规格（SXM）</a></li>
<li><span class="topic">H100 的 SM 数（132）</span>：NVIDIA，<a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA Hopper Architecture In-Depth</a>，2022</li>
<li><span class="topic">混合精度</span>：Micikevicius et al.，<a href="https://arxiv.org/abs/1710.03740">Mixed Precision Training</a>，ICLR 2018</li>
<li><span class="topic">BF16 训练</span>：Kalamkar et al.，<a href="https://arxiv.org/abs/1905.12322">A Study of BFLOAT16 for Deep Learning Training</a>，2019</li>
<li><span class="topic">activation checkpointing</span>：Chen et al.，<a href="https://arxiv.org/abs/1604.06174">Training Deep Nets with Sublinear Memory Cost</a>，2016</li>
<li><span class="topic">激活的估算式、Megatron-LM 的 SP</span>：Korthikanti et al.，<a href="https://arxiv.org/abs/2205.05198">Reducing Activation Recomputation in Large Transformer Models</a>，MLSys 2023</li>
<li><span class="topic">GPT-3 XL 的架构</span>：Brown et al.，<a href="https://arxiv.org/abs/2005.14165">Language Models are Few-Shot Learners</a>，NeurIPS 2020</li>
<li><span class="topic">各类算子的运算量与时间占比（BERT-large，V100）</span>：Ivanov et al.，<a href="https://arxiv.org/abs/2007.00072">Data Movement Is All You Need: A Case Study on Optimizing Transformers</a>，MLSys 2021</li>
<li><span class="topic">online softmax</span>：Milakov 与 Gimelshein，<a href="https://arxiv.org/abs/1805.02867">Online Normalizer Calculation for Softmax</a>，2018</li>
<li><span class="topic">FlashAttention</span>：Dao et al.，<a href="https://arxiv.org/abs/2205.14135">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</a>，NeurIPS 2022</li>
<li><span class="topic">Llama-3 405B 的参数、token 数、global batch、GPU 数、网络与并行配置</span>：Llama Team，<a href="https://arxiv.org/abs/2407.21783">The Llama 3 Herd of Models</a>，2024</li>
<li><span class="topic">ZeRO</span>：Rajbhandari et al.，<a href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a>，SC 2020</li>
<li><span class="topic">TP</span>：Shoeybi et al.，<a href="https://arxiv.org/abs/1909.08053">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism</a>，2019</li>
<li><span class="topic">CP（Li et al. 称为 sequence parallelism）</span>：Li et al.，<a href="https://arxiv.org/abs/2105.13120">Sequence Parallelism: Long Sequence Training from System Perspective</a>，2021</li>
<li><span class="topic">Ring Attention</span>：Liu et al.，<a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>，2023</li>
<li><span class="topic">DeepSpeed-Ulysses</span>：Jacobs et al.，<a href="https://arxiv.org/abs/2309.14509">DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models</a>，2023</li>
<li><span class="topic">TP、PP、DP 的组合与通信分析</span>：Narayanan et al.，<a href="https://arxiv.org/abs/2104.04473">Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</a>，SC 2021</li>
<li><span class="topic">PP 与 bubble</span>：Huang et al.，<a href="https://arxiv.org/abs/1811.06965">GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism</a>，NeurIPS 2019</li>
<li><span class="topic">critical batch size</span>：McCandlish et al.，<a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>，2018</li>
<li><span class="topic">scaling law</span>：Kaplan et al.，<a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a>，2020</li>
<li><span class="topic">MoE</span>：Shazeer et al.，<a href="https://arxiv.org/abs/1701.06538">Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer</a>，ICLR 2017</li>
<li><span class="topic">EP</span>：Lepikhin et al.，<a href="https://arxiv.org/abs/2006.16668">GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding</a>，ICLR 2021</li>
<li><span class="topic">DeepSeek-V3 的参数量与 EP 配置</span>：DeepSeek-AI，<a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report</a>，2024</li>
<li><span class="topic">Llama 2 7B 的层数与宽度（32 层，4096）</span>：Touvron et al.，<a href="https://arxiv.org/abs/2302.13971">LLaMA: Open and Efficient Foundation Language Models</a>，2023</li>
<li><span class="topic">Llama 2 7B 不用 GQA</span>：Touvron et al.，<a href="https://arxiv.org/abs/2307.09288">Llama 2: Open Foundation and Fine-Tuned Chat Models</a>，2023</li>
<li><span class="topic">continuous batching</span>：Yu et al.，<a href="https://www.usenix.org/conference/osdi22/presentation/yu">Orca: A Distributed Serving System for Transformer-Based Generative Models</a>，OSDI 2022</li>
<li><span class="topic">PagedAttention</span>：Kwon et al.，<a href="https://arxiv.org/abs/2309.06180">Efficient Memory Management for Large Language Model Serving with PagedAttention</a>，SOSP 2023</li>
<li><span class="topic">GQA</span>：Ainslie et al.，<a href="https://arxiv.org/abs/2305.13245">GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</a>，EMNLP 2023</li>
<li><span class="topic">MLA</span>：DeepSeek-AI，<a href="https://arxiv.org/abs/2405.04434">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</a>，2024</li>
<li><span class="topic">chunked prefill</span>：Agrawal et al.，<a href="https://arxiv.org/abs/2403.02310">Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve</a>，OSDI 2024</li>
<li><span class="topic">prefix caching</span>：Zheng et al.，<a href="https://arxiv.org/abs/2312.07104">SGLang: Efficient Execution of Structured Language Model Programs</a>，NeurIPS 2024</li>
<li><span class="topic">Flash-Decoding</span>：Dao et al.，<a href="https://pytorch.org/blog/flash-decoding/">Flash-Decoding for long-context inference（PyTorch 博客）</a>，2023</li>
<li><span class="topic">量化</span>：Frantar et al.，<a href="https://arxiv.org/abs/2210.17323">GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers</a>，ICLR 2023</li>
<li><span class="topic">speculative decoding</span>：Leviathan et al.，<a href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</a>，ICML 2023</li>
<li><span class="topic">P/D 分离</span>：Zhong et al.，<a href="https://arxiv.org/abs/2401.09670">DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving</a>，OSDI 2024</li>
<li><span class="topic">RL 的两个引擎与部署</span>：Sheng et al.，<a href="https://arxiv.org/abs/2409.19256">HybridFlow: A Flexible and Efficient RLHF Framework</a>，EuroSys 2025</li>
<li><span class="topic">rollout 提前开始的异步 RL</span>：Noukhovitch et al.，<a href="https://arxiv.org/abs/2410.18252">Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models</a>，ICLR 2025</li>
</ol>
