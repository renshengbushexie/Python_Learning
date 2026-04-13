from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = PROJECT_ROOT / "算法"
REFERENCE_SECTION_PATTERN = re.compile(r"^## 参考资料[\s\S]*$", re.MULTILINE)


@dataclass(frozen=True)
class DocPayload:
    tutorials: tuple[str, ...]
    notebooks: tuple[str, ...]
    story_lines: tuple[str, ...]
    d2l_notes: tuple[str, ...]
    selection_notes: tuple[str, ...]
    python_impl: str
    usage_example: str


TARGETS: dict[int, DocPayload] = {
    75: DocPayload(
        tutorials=(
            "[d2l 第4.6节 暂退法（Dropout）](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：dropout.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_multilayer-perceptrons/dropout.md)",
        ),
        story_lines=(
            "Dropout 的核心作用是抑制神经元之间的共适应，让模型在训练阶段就学会“部分特征缺失时也能工作”。",
            "它出现的直接背景是深层网络参数量快速增大后，训练集精度很高但验证集明显掉队，单靠早停和 L2 正则往往不够稳。",
            "和传统正则不同，Dropout 不是直接约束权重大小，而是通过随机屏蔽激活逼近“多个子网络集成”的效果。",
        ),
        d2l_notes=(
            "D2L 默认采用 inverted dropout，使训练阶段激活的期望保持不变，因此推理阶段无需再次缩放输出。",
            "教程同时给出从零实现和框架 API 实现，适合直接核对 `p` 的定义、掩码采样方式以及层放置位置。",
            "若与 BatchNorm 同用，应先固定无 Dropout 基线，再逐层增加 `p`，避免把归一化带来的波动误判为 Dropout 收益。",
        ),
        selection_notes=(
            "如果问题本质是模型容量过大而非数据泄漏，优先按 D2L 的做法在全连接层或预测头上尝试小到中等 `p`，再决定是否向前层扩展。",
            "如果加入 Dropout 后训练集和验证集同时变差，应先降低 `p` 或缩小模型，而不是继续叠加更多正则手段。",
        ),
        python_impl=dedent(
            """
            下面给出一个与 D2L 讲解一致的简化 PyTorch 实现，既保留手写 Dropout 层，也展示它如何嵌入 MLP。

            ```python
            import torch
            from torch import nn


            def dropout_layer(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
                if not training or p == 0.0:
                    return x
                if p >= 1.0:
                    return torch.zeros_like(x)
                mask = (torch.rand_like(x) > p).float()
                return mask * x / (1.0 - p)


            class MLPWithDropout(nn.Module):
                def __init__(self, in_features: int, hidden: int, out_features: int, p: float = 0.5):
                    super().__init__()
                    self.fc1 = nn.Linear(in_features, hidden)
                    self.act = nn.ReLU()
                    self.fc2 = nn.Linear(hidden, out_features)
                    self.p = p

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.fc1(x)
                    x = self.act(x)
                    x = dropout_layer(x, p=self.p, training=self.training)
                    return self.fc2(x)
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：在训练态与推理态分别前向一次，观察 Dropout 是否只在训练时生效。

            ```python
            import torch

            model = MLPWithDropout(in_features=8, hidden=32, out_features=2, p=0.3)
            batch = torch.randn(4, 8)

            model.train()
            train_logits = model(batch)

            model.eval()
            eval_logits = model(batch)

            print("train shape:", train_logits.shape)
            print("eval shape:", eval_logits.shape)
            print("same outputs:", torch.allclose(train_logits, eval_logits))
            ```
            """
        ).strip(),
    ),
    196: DocPayload(
        tutorials=(
            "[d2l 第11.10节 Adam算法](https://zh.d2l.ai/chapter_optimization/adam.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：adam.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_optimization/adam.md)",
        ),
        story_lines=(
            "Adam 的核心作用是同时利用一阶动量和二阶尺度信息，让优化器在稀疏梯度、噪声目标和复杂网络上都能较快进入可用区间。",
            "它出现的背景是：SGD+Momentum 往往需要更细致的学习率调参与更长热身，而 AdaGrad 又容易因为学习率衰减过快而后期停滞。",
            "因此 Adam 可以看作“动量法 + 自适应学习率”的折中方案，目标是降低冷启动成本并提升训练鲁棒性。",
        ),
        d2l_notes=(
            "D2L 将 Adam 解释为“动量法 + RMSProp + 偏差修正”的组合，这有助于把一阶矩、二阶矩与有效学习率联系起来理解。",
            "教程同时给出从零实现和框架接口实现，特别适合排查 `m_t`、`v_t` 张量形状、更新顺序和偏差修正是否写对。",
            "在工程验证时，除了看 loss 曲线，还应像 D2L 一样同时监控优化器状态和学习率调度，否则容易误判收敛速度差异的来源。",
        ),
        selection_notes=(
            "若目标是尽快得到可用基线，Adam/AdamW 仍然最省试错成本；若最终泛化更关键，应尽早做与 SGD/Momentum 的对照。",
            "当你观察到训练稳定但验证集迟迟不上升时，优先检查 `beta_2`、weight decay 和 scheduler，而不是盲目增大学习率。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版 Adam 优化器实现，用于帮助理解一阶矩、二阶矩和偏差修正是如何联动的。

            ```python
            import torch


            class AdamLike:
                def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
                    self.params = list(params)
                    self.lr = lr
                    self.beta1, self.beta2 = betas
                    self.eps = eps
                    self.weight_decay = weight_decay
                    self.step_count = 0
                    self.m = [torch.zeros_like(p) for p in self.params]
                    self.v = [torch.zeros_like(p) for p in self.params]

                @torch.no_grad()
                def step(self):
                    self.step_count += 1
                    for p, m, v in zip(self.params, self.m, self.v):
                        if p.grad is None:
                            continue
                        g = p.grad
                        if self.weight_decay:
                            g = g + self.weight_decay * p
                        m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                        v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                        m_hat = m / (1 - self.beta1 ** self.step_count)
                        v_hat = v / (1 - self.beta2 ** self.step_count)
                        p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)

                def zero_grad(self):
                    for p in self.params:
                        if p.grad is not None:
                            p.grad.zero_()
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：用手写 Adam 优化一元线性回归，确认损失能下降。

            ```python
            import torch
            from torch import nn

            x = torch.randn(128, 1)
            y = 3.0 * x + 0.5
            model = nn.Linear(1, 1)
            loss_fn = nn.MSELoss()
            optimizer = AdamLike(model.parameters(), lr=1e-2)

            for _ in range(200):
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print("final loss:", float(loss))
            ```
            """
        ).strip(),
    ),
    197: DocPayload(
        tutorials=(
            "[d2l 第11.7节 AdaGrad算法](https://zh.d2l.ai/chapter_optimization/adagrad.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：adagrad.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_optimization/adagrad.md)",
        ),
        story_lines=(
            "AdaGrad 的核心作用是为每个参数维度分配不同的学习率，让高频特征自动降步长、低频特征保持较大的更新幅度。",
            "它出现的直接背景是稀疏高维特征越来越常见，统一学习率往往会让常见特征更新过猛、罕见特征学不到东西。",
            "因此 AdaGrad 更像是“按坐标自适应”的优化器，它优先解决的是特征频次差异，而不是复杂非凸目标下的最终最优收敛。",
        ),
        d2l_notes=(
            "D2L 把 AdaGrad 的重点放在按坐标缩放学习率上，这能直接解释为何它对高维稀疏特征特别友好。",
            "教程中的累计平方梯度实现可直接映射到当前文档的 $G_t$ 公式，适合检查分母稳定项和初始累积器设置。",
            "如果你想复现 AdaGrad 的优势，最好像 D2L 一样在稀疏或凸优化任务上对比，而不是只在深度非凸任务上评价它。",
        ),
        selection_notes=(
            "在词袋、CTR、哈希特征这类高维稀疏问题上，可以优先沿用 D2L 的 AdaGrad 实现做 baseline，再视学习率过快衰减情况切到 RMSProp/Adam。",
            "若有效学习率很早就接近零，优先检查是否需要 `initial_accumulator_value` 或直接改用指数遗忘二阶矩优化器。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版 AdaGrad 实现，用累计平方梯度为每个参数维度调整步长。

            ```python
            import torch


            class AdaGradLike:
                def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
                    self.params = list(params)
                    self.lr = lr
                    self.eps = eps
                    self.weight_decay = weight_decay
                    self.accumulators = [torch.zeros_like(p) for p in self.params]

                @torch.no_grad()
                def step(self):
                    for p, acc in zip(self.params, self.accumulators):
                        if p.grad is None:
                            continue
                        g = p.grad
                        if self.weight_decay:
                            g = g + self.weight_decay * p
                        acc.addcmul_(g, g, value=1.0)
                        p.addcdiv_(g, acc.sqrt().add(self.eps), value=-self.lr)

                def zero_grad(self):
                    for p in self.params:
                        if p.grad is not None:
                            p.grad.zero_()
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：在稀疏输入特征上训练一个线性层，观察 AdaGrad 的稳定收敛。

            ```python
            import torch
            from torch import nn

            x = torch.randint(0, 2, (256, 20)).float()
            y = (x[:, :3].sum(dim=1, keepdim=True) > 1).float()
            model = nn.Linear(20, 1)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = AdaGradLike(model.parameters(), lr=0.2)

            for _ in range(150):
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print("final loss:", float(loss))
            ```
            """
        ).strip(),
    ),
    198: DocPayload(
        tutorials=(
            "[d2l 第11.8节 RMSProp算法](https://zh.d2l.ai/chapter_optimization/rmsprop.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：rmsprop.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_optimization/rmsprop.md)",
        ),
        story_lines=(
            "RMSProp 的核心作用是保留 AdaGrad 的按维度自适应思想，但避免学习率因为历史梯度无界累积而过早衰减到几乎无法更新。",
            "它出现的背景正是 AdaGrad 在长训练、非平稳目标和深度网络上经常“前期很快、后期停住”的问题。",
            "因此 RMSProp 可以理解为“带遗忘机制的 AdaGrad”，重点在于让优化器更关注近期梯度统计而不是全部历史。",
        ),
        d2l_notes=(
            "D2L 用指数加权移动平均替换 AdaGrad 的无界累积器，这正是 RMSProp 能在长训练中保持学习率活性的关键。",
            "教程里的平滑系数可以理解为“记忆窗口长度”，调它时应连同学习率一起看，而不是孤立调单个参数。",
            "RMSProp 常被用作序列模型和噪声较大目标上的稳定基线，实践中应像 D2L 那样关注每维尺度归一化后的更新幅度。",
        ),
        selection_notes=(
            "当 AdaGrad 衰减过快但 Adam 又显得过于激进时，RMSProp 往往是更稳的中间方案。",
            "如果验证曲线振荡明显，优先调平滑系数和学习率，再决定是否升级到 Adam 或加入动量变体。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版 RMSProp 实现，用指数滑动平均缓解 AdaGrad 学习率衰减过快的问题。

            ```python
            import torch


            class RMSPropLike:
                def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0):
                    self.params = list(params)
                    self.lr = lr
                    self.alpha = alpha
                    self.eps = eps
                    self.weight_decay = weight_decay
                    self.avg_sq = [torch.zeros_like(p) for p in self.params]

                @torch.no_grad()
                def step(self):
                    for p, avg in zip(self.params, self.avg_sq):
                        if p.grad is None:
                            continue
                        g = p.grad
                        if self.weight_decay:
                            g = g + self.weight_decay * p
                        avg.mul_(self.alpha).addcmul_(g, g, value=1 - self.alpha)
                        p.addcdiv_(g, avg.sqrt().add(self.eps), value=-self.lr)

                def zero_grad(self):
                    for p in self.params:
                        if p.grad is not None:
                            p.grad.zero_()
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：在带噪声的回归数据上训练一个线性层，观察 RMSProp 的收敛效果。

            ```python
            import torch
            from torch import nn

            x = torch.randn(512, 10)
            y = x[:, :1] * 2.5 - 0.8 + 0.1 * torch.randn(512, 1)
            model = nn.Linear(10, 1)
            loss_fn = nn.MSELoss()
            optimizer = RMSPropLike(model.parameters(), lr=1e-2, alpha=0.95)

            for _ in range(200):
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print("final loss:", float(loss))
            ```
            """
        ).strip(),
    ),
    98: DocPayload(
        tutorials=(
            "[d2l 第15.8节 BERT](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert.html)",
            "[d2l 第15.9节 用于预训练BERT的数据集](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html)",
            "[d2l 第15.10节 预训练BERT](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert-pretraining.html)",
            "[d2l 第15.11节 微调BERT](https://zh.d2l.ai/chapter_natural-language-processing-applications/finetuning-bert.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：bert.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_natural-language-processing-pretraining/bert.md)",
            "[d2l-zh 对应笔记：bert-dataset.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_natural-language-processing-pretraining/bert-dataset.md)",
            "[d2l-zh 对应笔记：bert-pretraining.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_natural-language-processing-pretraining/bert-pretraining.md)",
            "[d2l-zh 对应笔记：finetuning-bert.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_natural-language-processing-applications/finetuning-bert.md)",
        ),
        story_lines=(
            "BERT 的核心作用，是先用海量无标注文本学到可迁移的双向上下文表示，再把这套表示迁移到分类、匹配、抽取等下游任务上。",
            "它出现的背景是：传统词向量只能给出静态词表示，而很多 NLP 任务真正需要的是“同一个词在不同语境里的不同含义”。",
            "相比单向语言模型，BERT 通过掩码语言模型在训练时同时看左右文，目标就是让编码器更适合判别式自然语言理解任务。",
        ),
        d2l_notes=(
            "D2L 把 BERT 拆成“输入表示、预训练数据构造、预训练目标、下游微调”四步来讲，这很适合把概念和工程流程一一对齐。",
            "教程明确展示了 `[CLS]`、`[SEP]`、segment embedding 和 15% mask 采样规则，适合直接对应当前文档中的结构说明。",
            "如果你要把 BERT 思想迁移到其他编码器模型，优先保留“预训练再微调”范式，而不是机械照搬 NSP 等原始细节。",
        ),
        selection_notes=(
            "文本分类、序列标注、抽取式问答这类判别任务，仍然适合先从 BERT 或其改进版本起步。",
            "如果任务是自由生成、长文续写或工具调用，优先考虑解码器或编码器-解码器架构，而不是硬把 BERT 改成生成模型。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版 `MiniBERTClassifier`，用 PyTorch 原生模块把 token/position/segment embedding、编码器和分类头串起来。

            ```python
            import torch
            from torch import nn


            class MiniBERTClassifier(nn.Module):
                def __init__(
                    self,
                    vocab_size: int,
                    hidden_size: int = 128,
                    num_layers: int = 2,
                    num_heads: int = 4,
                    max_len: int = 128,
                    num_classes: int = 2,
                ):
                    super().__init__()
                    self.token_embedding = nn.Embedding(vocab_size, hidden_size)
                    self.position_embedding = nn.Embedding(max_len, hidden_size)
                    self.segment_embedding = nn.Embedding(2, hidden_size)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        dropout=0.1,
                        batch_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.classifier = nn.Linear(hidden_size, num_classes)

                def forward(
                    self,
                    token_ids: torch.Tensor,
                    segment_ids: torch.Tensor,
                    attention_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
                    batch_size, seq_len = token_ids.shape
                    positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
                    x = (
                        self.token_embedding(token_ids)
                        + self.position_embedding(positions)
                        + self.segment_embedding(segment_ids)
                    )
                    padding_mask = None
                    if attention_mask is not None:
                        padding_mask = attention_mask == 0
                    encoded = self.encoder(x, src_key_padding_mask=padding_mask)
                    cls_repr = encoded[:, 0]
                    return self.classifier(cls_repr)
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：构造一批伪 token 序列，执行一次 BERT 风格分类前向和损失计算。

            ```python
            import torch
            from torch import nn

            model = MiniBERTClassifier(vocab_size=30522, hidden_size=64, num_layers=2, num_heads=4, max_len=32)
            token_ids = torch.randint(0, 30522, (4, 16))
            segment_ids = torch.zeros_like(token_ids)
            attention_mask = torch.ones_like(token_ids)
            labels = torch.tensor([0, 1, 1, 0])

            logits = model(token_ids, segment_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            print("logits shape:", logits.shape)
            print("loss:", float(loss))
            ```
            """
        ).strip(),
    ),
    104: DocPayload(
        tutorials=(
            "[d2l 第9.5节 长短期记忆网络（LSTM）](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：lstm.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_recurrent-modern/lstm.md)",
        ),
        story_lines=(
            "LSTM 的核心作用，是在序列建模里显式保留一条较稳定的记忆通道，让模型能够跨越更多时间步保留关键信息。",
            "它出现的背景，是普通 RNN 在长序列训练中容易发生梯度消失或爆炸，导致模型明明看到了历史，却学不会真正的长期依赖。",
            "LSTM 通过输入门、遗忘门和输出门把“写入、保留、读取”三件事分开处理，本质上是在给循环网络增加可控记忆机制。",
        ),
        d2l_notes=(
            "D2L 对 LSTM 的讲解重点是记忆单元 `c_t` 与门控之间的关系，这一点正好对应当前文档里“为何能缓解梯度消失”的核心问题。",
            "教程同时给出从零实现和框架版实现，适合核对门控公式、状态维度以及批量序列输入的组织方式。",
            "把 D2L 代码迁移到时间序列预测时，通常只需把语言模型的词嵌入输入换成数值特征序列，再接一个回归头即可。",
        ),
        selection_notes=(
            "如果业务数据存在明显长期依赖、序列长度中等且数据量不算特别大，LSTM 依旧是很稳的深度时序基线。",
            "如果需要更强并行性、超长上下文或多变量跨周期建模能力，再进一步比较 Transformer 类方法更合适。",
        ),
        python_impl=dedent(
            """
            下面给出一个用于多变量时间序列预测的简化 LSTM 实现，输入形状为 `(batch, seq_len, feature_dim)`。

            ```python
            import torch
            from torch import nn


            class LSTMForecaster(nn.Module):
                def __init__(
                    self,
                    input_size: int,
                    hidden_size: int = 64,
                    num_layers: int = 2,
                    horizon: int = 1,
                ):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=0.1 if num_layers > 1 else 0.0,
                        batch_first=True,
                    )
                    self.head = nn.Linear(hidden_size, horizon)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    output, _ = self.lstm(x)
                    last_hidden = output[:, -1]
                    return self.head(last_hidden)
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：构造一批正弦序列片段，用 LSTM 预测下一步值。

            ```python
            import torch

            model = LSTMForecaster(input_size=3, hidden_size=32, num_layers=2, horizon=1)
            batch = torch.randn(8, 24, 3)
            target = torch.randn(8, 1)

            prediction = model(batch)
            loss = torch.mean((prediction - target) ** 2)

            print("prediction shape:", prediction.shape)
            print("loss:", float(loss))
            ```
            """
        ).strip(),
    ),
    105: DocPayload(
        tutorials=(
            "[d2l 第10.3节 自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)",
            "[d2l 第10.7节 Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：self-attention-and-positional-encoding.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/self-attention-and-positional-encoding.md)",
            "[d2l-zh 对应笔记：transformer.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/transformer.md)",
        ),
        story_lines=(
            "Transformer 时间序列模型的核心作用，是用自注意力直接建立任意两个时间步之间的关系，从而更高效地建模长距离依赖。",
            "它出现的背景是：RNN/LSTM 虽然能处理序列，但训练天然串行、长依赖路径长，在长窗口预测和大规模并行训练时越来越吃力。",
            "把 Transformer 引入时间序列，本质上是把“词元之间的依赖建模”迁移成“时间步之间的依赖建模”，再为连续值预测补上合适的输入投影与输出头。",
        ),
        d2l_notes=(
            "D2L 先讲自注意力和位置编码，再讲完整 Transformer，这条路径很适合迁移到时序建模：先理解依赖关系，再理解整体块结构。",
            "教程里的位置编码思想可以直接复用到时间序列；差别主要在于输入不再是离散 token，而是连续时间步特征。",
            "工程上很多时序 Transformer 只保留编码器部分，并把最后若干时间步表示映射到预测窗口，这一点可以作为从 D2L 标准结构到时序版本的最短迁移路径。",
        ),
        selection_notes=(
            "当输入窗口很长、变量之间有明显跨周期关联、同时还想保留较好的训练并行性时，优先评估 Transformer 类时间序列模型。",
            "如果样本很少、解释需求更强或部署资源紧张，先用统计模型或 LSTM 建基线通常更划算。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版时间序列 Transformer，实现“连续特征输入 + 位置编码 + 编码器 + 多步预测头”。

            ```python
            import math
            import torch
            from torch import nn


            class PositionalEncoding(nn.Module):
                def __init__(self, d_model: int, max_len: int = 512):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(max_len).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    self.register_buffer("pe", pe.unsqueeze(0))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x + self.pe[:, : x.size(1)]


            class TemporalTransformer(nn.Module):
                def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, horizon: int = 1):
                    super().__init__()
                    self.input_proj = nn.Linear(input_size, d_model)
                    self.position = PositionalEncoding(d_model=d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 4,
                        dropout=0.1,
                        batch_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.head = nn.Linear(d_model, horizon)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.input_proj(x)
                    x = self.position(x)
                    encoded = self.encoder(x)
                    return self.head(encoded[:, -1])
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：输入一段多变量历史窗口，输出未来两个时间步的预测值。

            ```python
            import torch

            model = TemporalTransformer(input_size=5, d_model=64, nhead=4, num_layers=2, horizon=2)
            batch = torch.randn(16, 48, 5)
            prediction = model(batch)

            print("prediction shape:", prediction.shape)
            ```
            """
        ).strip(),
    ),
    201: DocPayload(
        tutorials=(
            "[d2l 第12.1节 异步计算](https://zh.d2l.ai/chapter_computational-performance/async-computation.html)",
            "[d2l 第12.5节 参数服务器](https://zh.d2l.ai/chapter_computational-performance/parameterserver.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：async-computation.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computational-performance/async-computation.md)",
            "[d2l-zh 对应笔记：parameterserver.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computational-performance/parameterserver.md)",
        ),
        story_lines=(
            "参数服务器的核心作用，是把模型参数的存储与更新从 worker 计算节点里拆出来，用专门的参数分片服务承接 push/pull 交互。",
            "它出现的背景，是单机内存和单节点带宽都不再足以支撑超大模型训练，尤其是推荐系统和稀疏 embedding 类任务对参数规模极其敏感。",
            "相比纯同步 AllReduce，参数服务器强调的是“参数集中管理 + 分片扩展 + 异构容忍”，因此更适合复杂集群和稀疏更新场景。",
        ),
        d2l_notes=(
            "D2L 先用异步计算讲清“计算与通信可以并行”，再用参数服务器解释 push/pull 架构，这是理解分布式训练系统设计的很好入口。",
            "教程重点不在复杂公式，而在角色拆分：worker 负责算梯度，server 负责存参数和应用更新，这正好对应当前条目的系统视角。",
            "把 D2L 的图示迁移到工程实现时，最需要补充的是参数分片、热点键路由和一致性策略，这些是生产级 PS 的落地点。",
        ),
        selection_notes=(
            "稀疏参数量巨大、节点异构明显、训练任务希望容忍慢节点时，参数服务器通常比纯 AllReduce 更合适。",
            "如果是稠密模型、GPU 集群同构且能严格同步，先比较 AllReduce 往往能拿到更好的带宽利用率。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版参数服务器模拟：server 管理参数分片，worker 拉取参数、计算梯度并回推更新。

            ```python
            import torch


            class ToyParameterServer:
                def __init__(self, learning_rate: float = 1e-2):
                    self.learning_rate = learning_rate
                    self.shards = {
                        "weight": torch.zeros(1, 1),
                        "bias": torch.zeros(1),
                    }

                def pull(self) -> dict[str, torch.Tensor]:
                    return {name: tensor.clone() for name, tensor in self.shards.items()}

                def push(self, grads: dict[str, torch.Tensor]) -> None:
                    for name, grad in grads.items():
                        self.shards[name] -= self.learning_rate * grad


            class LinearWorker:
                def __init__(self, server: ToyParameterServer):
                    self.server = server

                def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
                    params = self.server.pull()
                    weight = params["weight"].requires_grad_()
                    bias = params["bias"].requires_grad_()
                    pred = x @ weight + bias
                    loss = torch.mean((pred - y) ** 2)
                    loss.backward()
                    self.server.push({"weight": weight.grad, "bias": bias.grad})
                    return float(loss)
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：两个 worker 轮流在同一个参数服务器上训练线性回归。

            ```python
            import torch

            server = ToyParameterServer(learning_rate=0.05)
            workers = [LinearWorker(server), LinearWorker(server)]
            x = torch.randn(64, 1)
            y = 2.0 * x + 0.3

            for step in range(20):
                worker = workers[step % len(workers)]
                loss = worker.train_step(x, y)

            print("final weight:", server.shards["weight"].item())
            print("final bias:", server.shards["bias"].item())
            print("last loss:", loss)
            ```
            """
        ).strip(),
    ),
    204: DocPayload(
        tutorials=(
            "[d2l 第12.1节 异步计算](https://zh.d2l.ai/chapter_computational-performance/async-computation.html)",
            "[d2l 第12.5节 参数服务器](https://zh.d2l.ai/chapter_computational-performance/parameterserver.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：async-computation.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computational-performance/async-computation.md)",
            "[d2l-zh 对应笔记：parameterserver.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computational-performance/parameterserver.md)",
        ),
        story_lines=(
            "异步 SGD 的核心作用，是让不同 worker 不必在每一步都等待彼此对齐，而是谁先算完就先提交梯度，从而提升整体吞吐。",
            "它出现的背景，是同步训练在大规模集群里经常被慢节点拖住，导致大量计算资源空等屏障。",
            "异步 SGD 的本质权衡很明确：用更高的硬件利用率换取一定的梯度陈旧和收敛噪声，因此它首先是一种系统优化策略，其次才是一种优化器变体。",
        ),
        d2l_notes=(
            "D2L 在讲异步计算时强调“GPU 计算和通信并不一定要串行执行”，这正是异步 SGD 能带来吞吐提升的底层原因。",
            "教程中的异步调度思路可以直接迁移到训练系统理解里：worker 先工作，参数更新稍后汇合，而不是每步都全局 barrier。",
            "如果你在工程里实现异步 SGD，除了看 loss 曲线，还要像 D2L 一样关注流水线是否真的被填满，否则异步复杂度可能得不偿失。",
        ),
        selection_notes=(
            "当目标是更快得到可用模型、集群异构明显且对最终最优精度容忍轻微损失时，异步 SGD 值得优先验证。",
            "如果任务追求稳定复现和最佳最终指标，半同步或同步训练通常更容易控制实验噪声。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版异步 SGD 模拟器，用“延迟队列”近似不同 worker 的陈旧梯度提交。

            ```python
            import random
            import torch


            class AsyncSGDSimulator:
                def __init__(self, dim: int, lr: float = 1e-2):
                    self.weight = torch.zeros(dim, 1)
                    self.lr = lr
                    self.pending: list[tuple[int, torch.Tensor]] = []

                def submit(self, grad: torch.Tensor, delay: int) -> None:
                    self.pending.append((delay, grad.clone()))

                def tick(self) -> None:
                    next_pending: list[tuple[int, torch.Tensor]] = []
                    for delay, grad in self.pending:
                        if delay <= 0:
                            self.weight -= self.lr * grad
                        else:
                            next_pending.append((delay - 1, grad))
                    self.pending = next_pending


            def compute_linear_grad(weight: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pred = x @ weight
                grad = (2.0 / x.size(0)) * x.T @ (pred - y)
                return grad
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：3 个 worker 以不同延迟把梯度推给同一个参数向量。

            ```python
            import random
            import torch

            sim = AsyncSGDSimulator(dim=4, lr=0.05)
            x = torch.randn(32, 4)
            true_w = torch.tensor([[2.0], [-1.0], [0.5], [0.8]])
            y = x @ true_w

            for _ in range(30):
                stale_weight = sim.weight.clone()
                grad = compute_linear_grad(stale_weight, x, y)
                sim.submit(grad, delay=random.randint(0, 2))
                sim.tick()

            print("estimated weight:", sim.weight.squeeze())
            ```
            """
        ).strip(),
    ),
    225: DocPayload(
        tutorials=(
            "[d2l 第10.3节 自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)",
            "[d2l 第10.7节 Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：self-attention-and-positional-encoding.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/self-attention-and-positional-encoding.md)",
            "[d2l-zh 对应笔记：transformer.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/transformer.md)",
        ),
        story_lines=(
            "时空 Transformer 的核心作用，是同时建模“时间上的远距离依赖”和“空间上的跨节点交互”，让模型不再局限于局部卷积或短程递归。",
            "它出现的背景，是 ConvLSTM、时空卷积和图卷积虽然有效，但往往带有较强局部归纳偏置，面对长时间跨度和远距离空间耦合时容易受限。",
            "从方法论上看，时空 Transformer 是把标准 Transformer 的 token 依赖建模扩展到“时间步 × 空间位置”的二维关系建模。",
        ),
        d2l_notes=(
            "D2L 并未直接给出时空 Transformer，但自注意力、位置编码和 Transformer 模块已经提供了最关键的骨架。",
            "把 D2L 的标准序列 token 替换成“节点-时间片 token”，再补上空间位置编码、图先验或局部窗口限制，就能得到时空版本。",
            "因此学习顺序很清晰：先吃透 D2L 的注意力机制，再理解时空建模里为什么要加入空间结构偏置。",
        ),
        selection_notes=(
            "当任务既有长时间依赖又有明显空间关联，例如交通、气象、人体骨架动作识别时，时空 Transformer 很值得作为强基线。",
            "如果空间拓扑非常稳定且局部性特别强，先比较 STGCN、ConvLSTM 等归纳偏置更强的方法，通常更省数据也更省算力。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版时空 Transformer：先把 `(时间, 节点)` 组合展平成 token，再做编码并预测下一时刻各节点值。

            ```python
            import torch
            from torch import nn


            class SpatiotemporalTransformer(nn.Module):
                def __init__(self, node_count: int, feature_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
                    super().__init__()
                    self.node_embedding = nn.Embedding(node_count, d_model)
                    self.time_projection = nn.Linear(feature_dim, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 4,
                        dropout=0.1,
                        batch_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.head = nn.Linear(d_model, 1)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    batch_size, seq_len, node_count, feature_dim = x.shape
                    node_ids = torch.arange(node_count, device=x.device)
                    node_embed = self.node_embedding(node_ids).view(1, 1, node_count, -1)
                    token = self.time_projection(x) + node_embed
                    token = token.view(batch_size, seq_len * node_count, -1)
                    encoded = self.encoder(token)
                    last_frame = encoded.view(batch_size, seq_len, node_count, -1)[:, -1]
                    return self.head(last_frame).squeeze(-1)
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：输入过去 12 个时间步、6 个空间节点的特征，输出下一时刻每个节点的预测值。

            ```python
            import torch

            model = SpatiotemporalTransformer(node_count=6, feature_dim=3, d_model=64, nhead=4, num_layers=2)
            batch = torch.randn(4, 12, 6, 3)
            prediction = model(batch)

            print("prediction shape:", prediction.shape)  # (batch, node_count)
            ```
            """
        ).strip(),
    ),
    301: DocPayload(
        tutorials=(
            "[d2l 第10.3节 自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)",
            "[d2l 第10.7节 Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)",
        ),
        notebooks=(
            "[d2l-zh 对应笔记：self-attention-and-positional-encoding.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/self-attention-and-positional-encoding.md)",
            "[d2l-zh 对应笔记：transformer.md](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_attention-mechanisms/transformer.md)",
        ),
        story_lines=(
            "MoE-Transformer 的核心作用，是在不线性增加每个 token 计算量的前提下，把模型参数容量大幅扩展到更高规模。",
            "它出现的背景，是稠密 Transformer 扩参虽然有效，但训练和推理 FLOPs、显存和通信成本会随参数规模同步上涨，很快触到工程上限。",
            "MoE 的思路是让不同 token 只激活少数专家网络，把“模型更大”与“每次都算全部参数”拆开，这就是它成为大模型扩展路线的重要原因。",
        ),
        d2l_notes=(
            "D2L 没有直接覆盖 MoE，但 Transformer 的注意力、前馈网络和 token 处理流程已经把基础骨架讲得很完整。",
            "把 D2L 的标准前馈块替换为“路由器 + 多专家前馈层”，就能得到 MoE 版 Transformer 的核心形态。",
            "因此学习 MoE 最好的顺序依然是先吃透 D2L 的标准 Transformer，再理解为什么要把 dense FFN 变成 sparse expert routing。",
        ),
        selection_notes=(
            "当训练预算足够、数据分布多样、并且希望继续扩大参数容量但不愿让每 token FLOPs 同步飙升时，MoE 值得重点评估。",
            "如果数据规模有限、基础设施不支持专家并行或通信代价太高，先用稠密 Transformer 做扎实基线通常更稳。",
        ),
        python_impl=dedent(
            """
            下面给出一个教学版 Top-1 MoE 前馈层，实现“路由器挑专家 + 专家独立计算 + 输出回填”。

            ```python
            import torch
            from torch import nn


            class Top1MoE(nn.Module):
                def __init__(self, d_model: int, expert_hidden: int, num_experts: int):
                    super().__init__()
                    self.router = nn.Linear(d_model, num_experts)
                    self.experts = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(d_model, expert_hidden),
                                nn.GELU(),
                                nn.Linear(expert_hidden, d_model),
                            )
                            for _ in range(num_experts)
                        ]
                    )

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    route_logits = self.router(x)
                    route_index = route_logits.argmax(dim=-1)
                    output = torch.zeros_like(x)
                    for expert_id, expert in enumerate(self.experts):
                        mask = route_index == expert_id
                        if mask.any():
                            output[mask] = expert(x[mask])
                    return output, route_index
            ```
            """
        ).strip(),
        usage_example=dedent(
            """
            最小示例：输入一批 token 表示，观察每个 token 被路由到哪个专家。

            ```python
            import torch

            moe = Top1MoE(d_model=32, expert_hidden=64, num_experts=4)
            token_states = torch.randn(2, 8, 32)
            transformed, expert_ids = moe(token_states)

            print("output shape:", transformed.shape)
            print("expert ids shape:", expert_ids.shape)
            print("expert ids:", expert_ids)
            ```
            """
        ).strip(),
    ),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def find_doc_path(algo_id: int) -> Path:
    matches = sorted(ALGORITHM_ROOT.rglob(f"{algo_id:03d}-*.md"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file for id {algo_id:03d}, got {len(matches)}")
    return matches[0]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def upsert_subsection(text: str, section_heading: str, subsection_heading: str, bullet_lines: tuple[str, ...]) -> str:
    section_pattern = re.compile(rf"(?ms)^## {re.escape(section_heading)}\n(?P<body>.*?)(?=^## |\Z)")
    match = section_pattern.search(text)
    if not match:
        return text
    body = match.group("body").rstrip()
    subsection_pattern = re.compile(rf"(?ms)^### {re.escape(subsection_heading)}\n.*?(?=^### |\Z)")
    body = subsection_pattern.sub("", body).rstrip()
    addition = "\n".join([f"### {subsection_heading}", *[f"- {line}" for line in bullet_lines]])
    new_body = f"{body}\n\n{addition}\n\n"
    return text[: match.start('body')] + new_body + text[match.end('body') :]


def upsert_full_section_before_reference(text: str, section_heading: str, body: str) -> str:
    section_pattern = re.compile(rf"(?ms)^## {re.escape(section_heading)}\n.*?(?=^## |\Z)")
    text = section_pattern.sub("", text).rstrip() + "\n"
    new_section = f"\n## {section_heading}\n{body.strip()}\n"
    ref_match = re.search(r"^## 参考资料", text, re.MULTILINE)
    if ref_match:
        return text[: ref_match.start()].rstrip() + new_section + "\n\n" + text[ref_match.start() :].lstrip()
    return text.rstrip() + new_section + "\n"


def parse_reference_groups(text: str) -> dict[str, list[str]]:
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return {}
    groups: dict[str, list[str]] = {}
    current_heading: str | None = None
    for raw_line in match.group(0).splitlines()[1:]:
        line = raw_line.strip()
        if line.startswith("### "):
            current_heading = line[4:].strip()
            groups.setdefault(current_heading, [])
            continue
        if line.startswith("- ") and current_heading:
            groups[current_heading].append(line[2:].strip())
    return groups


def rewrite_reference_section(text: str, tutorials: tuple[str, ...], notebooks: tuple[str, ...]) -> str:
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return text
    groups = parse_reference_groups(text)
    textbooks = dedupe_preserve_order(list(tutorials) + groups.get("教材与综述", []) + groups.get("教材参考", []))
    extended = dedupe_preserve_order(list(notebooks) + groups.get("扩展阅读", []))
    passthrough = {
        heading: dedupe_preserve_order(items)
        for heading, items in groups.items()
        if heading not in {"教材与综述", "教材参考", "扩展阅读"}
    }
    prefix = text[: match.start()].rstrip()
    lines = ["## 参考资料", "### 教材与综述"]
    lines.extend(f"- {item}" for item in textbooks)
    for heading, items in passthrough.items():
        if not items:
            continue
        lines.extend(["", f"### {heading}"])
        lines.extend(f"- {item}" for item in items)
    if extended:
        lines.extend(["", "### 扩展阅读"])
        lines.extend(f"- {item}" for item in extended)
    return prefix + "\n\n" + "\n".join(lines) + "\n"


def enrich_doc(path: Path, payload: DocPayload) -> bool:
    original = read_text(path)
    updated = original
    updated = upsert_subsection(updated, "适用场景", "作用与产生背景", payload.story_lines)
    updated = upsert_subsection(
        updated,
        "模型实现原理",
        "D2L 对照学习",
        (
            f"对应教程：{'；'.join(payload.tutorials)}。",
            f"代码入口：{'；'.join(payload.notebooks)}。",
            *payload.d2l_notes,
        ),
    )
    updated = upsert_subsection(updated, "选型建议", "D2L 迁移提示", payload.selection_notes)
    updated = upsert_full_section_before_reference(updated, "Python代码实现", payload.python_impl)
    updated = upsert_full_section_before_reference(updated, "使用示例", payload.usage_example)
    updated = rewrite_reference_section(updated, payload.tutorials, payload.notebooks)
    if updated != original:
        write_text(path, updated)
        return True
    return False


def main() -> int:
    changed: list[Path] = []
    for algo_id, payload in TARGETS.items():
        path = find_doc_path(algo_id)
        if enrich_doc(path, payload):
            changed.append(path)

    print(f"D2L enhanced docs: {len(changed)}")
    for path in changed:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
