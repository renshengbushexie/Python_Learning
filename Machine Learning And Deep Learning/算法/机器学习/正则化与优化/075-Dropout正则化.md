# 075. Dropout正则化（Dropout）

> 一级分类：机器学习
> 二级分类：正则化与优化
> 任务类型：训练技巧
> 原始来源：readme_learning01.md / 正则化与优化技术

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| Dropout正则化 | 训练技巧 | 深层网络防过拟合；小样本训练；鲁棒特征学习 | 实现简单且常有效；近似模型集成效果 | 训练收敛变慢；与BN叠加时需谨慎放置 | 分层设置drop率；结合学习率计划和权重衰减 | drop probability $p$、层位置、是否inverted dropout | 数据量有限或模型容量大时优先开启 | 对比有/无dropout的泛化差距与激活稀疏度 |

## 适用场景
- 参数量大于样本量、验证集明显过拟合的神经网络。
- NLP与推荐中embedding后全连接层正则化。
- 需快速构建鲁棒基线的通用深度学习任务。

## 模型实现原理
对某层激活$h\in\mathbb{R}^d$采样掩码
$$
m_i\sim \mathrm{Bernoulli}(1-p),\quad \tilde{h}=m\odot h.
$$
inverted dropout训练时使用
$$
\tilde{h}=\frac{m}{1-p}\odot h,
$$
从而
$$
\mathbb{E}[\tilde{h}_i]=\mathbb{E}\left[\frac{m_i}{1-p}h_i\right]=h_i,
$$
测试时无需再缩放。

线性层$y=w^T\tilde{h}$的方差为
$$
\mathrm{Var}(y)=\sum_i w_i^2 h_i^2\frac{p}{1-p},
$$
可视为向激活注入与幅度相关的噪声，抑制神经元共适应。

从模型集成角度，Dropout近似对$2^d$个子网络做参数共享平均。

## 主要优点
- 有效降低训练-验证性能差距，泛化提升明显。
- 不增加可学习参数，改造成本低。
- 对输入扰动和局部缺失具有一定鲁棒性。

## 主要缺点
- 随机屏蔽引入梯度噪声，收敛速度下降。
- 过大$p$会导致表示能力不足并出现欠拟合。

## 针对缺点的解决办法
- 前层用小drop率、后层用较大drop率，分层调节噪声强度。
- 结合余弦退火/预热学习率，缓解收敛变慢。
- 与BatchNorm同用时通常将Dropout放在激活后且靠近分类头。

## 常用参数
- `p`：丢弃概率（常见0.1~0.5）。
- `dropout_position`：层前/层后与残差分支位置。
- `schedule`：固定或随epoch变化的drop率。
- `mc_dropout`：推理时是否保留dropout用于不确定性。

## 选型建议
- 当模型容量明显过大、数据中等偏小时优先使用Dropout。
- 对大规模预训练模型微调，先小$p$试验再决定是否保留。

## 模型解释方法
- 对比train/eval激活分布，确认噪声注入强度是否合理。
- 监控不同$p$下的校准误差(ECE)与泛化间隙。
- 统计神经元失活频率，避免关键通道长期被屏蔽。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
- [Dropout教程](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
