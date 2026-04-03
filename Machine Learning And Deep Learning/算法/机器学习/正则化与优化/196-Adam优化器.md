# 196. Adam优化器（Adam Optimization）

> 一级分类：机器学习
> 二级分类：正则化与优化
> 任务类型：优化算法
> 原始来源：readme_learning02.md / 优化算法

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| Adam优化器 | 优化算法 | 稀疏梯度训练；大规模深度学习；非平稳目标 | 一阶二阶矩自适应，收敛快且鲁棒 | 泛化有时弱于SGD；额外状态内存开销大 | 用AdamW解耦权重衰减；后期切换SGD或降学习率 | $\eta,\beta_1,\beta_2,\epsilon$、weight decay | 训练初期与复杂网络默认优先 | 观察一二阶矩、有效学习率、参数更新范数 |

## 适用场景
- Transformer、扩散模型等参数规模大且梯度统计复杂的训练。
- 稀疏特征或噪声较大目标函数优化。
- 快速得到可用基线并缩短收敛前期时间。

## 模型实现原理
给定梯度$g_t=\nabla_\theta \mathcal{L}(\theta_{t-1})$，Adam维护指数滑动平均：
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
$$
$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2.
$$
因初始化$ m_0=v_0=0 $带偏差，作修正：
$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}.
$$
参数更新
$$
\theta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
$$
可视为每个维度自适应步长
$$
\eta_{t,i}^{\text{eff}}=\frac{\eta}{\sqrt{\hat{v}_{t,i}}+\epsilon},
$$
梯度方差大的维度步长更小。AdamW将权重衰减与梯度更新解耦：
$$
\theta_t\leftarrow \theta_t-\eta\lambda\theta_{t-1}.
$$

## 主要优点
- 对学习率初值不敏感，通常较快达到可用精度。
- 对稀疏梯度和非平稳目标表现稳定。
- 减少手工调度负担，工程起步成本低。

## 主要缺点
- 额外维护$m,v$两组状态，显存约增2倍参数量。
- 某些任务最终泛化不及SGD+Momentum。
- $\beta_2$过大可能导致二阶矩响应滞后。

## 针对缺点的解决办法
- 使用AdamW替代Adam避免L2与自适应步长耦合问题。
- 后期切换到SGD或降低$\eta$做精细收敛。
- 混合精度下结合梯度裁剪，防止极端步长更新。

## 常用参数
- `lr(\eta)`：基础学习率。
- `betas=(\beta_1,\beta_2)`：动量与二阶矩平滑。
- `eps`：数值稳定项。
- `weight_decay`：建议采用AdamW解耦形式。

## 选型建议
- 新任务冷启动默认Adam/AdamW，快速到达可用区间。
- 对最终泛化敏感的视觉分类可尝试后期切SGD。

## 模型解释方法
- 记录$\|m_t\|,\|v_t\|$与有效学习率分布。
- 监控参数更新/参数本身比值$\|\Delta\theta\|/\|\theta\|$。
- 对比Adam与AdamW同配置下验证曲线和校准误差。

## 参考资料
- [Legacy Source: readme_learning02.md](../../../归档/readme_learning02.md)
