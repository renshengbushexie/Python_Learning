# 105. Transformer时间序列（Transformer）

> 一级分类：机器学习
> 二级分类：时间序列
> 任务类型：时间序列预测
> 原始来源：readme_learning01.md / 时间序列算法

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| Transformer时间序列 | 时间序列预测 | 长输入窗口、多变量高维时序、跨周期依赖显著的预测任务 | 并行训练高效；自注意力擅长长程依赖；易扩展多头多尺度结构 | 注意力复杂度高；数据需求大；小样本易过拟合 | 稀疏注意力（Informer）/分解结构（Autoformer）降复杂度；正则化与预训练 | `d_model`, `n_heads`, `n_layers`, `dropout`, `context_len` | 长序列预测优先评估；中短序列可先用LSTM/ARIMA做成本对照 | 注意力图、分解项可视化、输入步贡献分析 |

## 适用场景
- 电力负荷、交通流量等需要长历史窗口的多步预测。
- 多传感器工业监控中跨变量延迟影响明显的场景。
- 金融高维因子序列中跨周期关联建模与风险预警。

## 模型实现原理
给定嵌入后序列矩阵 $X\in\mathbb{R}^{T\times d}$，线性映射得到
$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$$ 
单头自注意力：
$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$ 
多头拼接：
$$\mathrm{MHA}(X)=\mathrm{Concat}(H_1,\ldots,H_h)W_O.$$ 
为保留时间顺序，加入位置编码：
$$PE_{pos,2i}=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad
PE_{pos,2i+1}=\cos\left(\frac{pos}{10000^{2i/d}}\right).$$

预测头可写为 $\hat Y=f_\theta(X_{1:T})$，最小化
$$\mathcal L=\frac{1}{H}\sum_{h=1}^{H}(y_{T+h}-\hat y_{T+h})^2.$$ 
标准注意力复杂度为 $O(T^2d)$。Informer 通过 ProbSparse 将复杂度降至近似 $O(T\log T)$；Autoformer 先分解趋势/季节，再在子序列上执行自相关注意力，提高长序列稳定性。

## 主要优点
- 对远距离依赖建模能力强，适合长预测跨度任务。
- 训练并行化程度高，吞吐优于递归RNN结构。
- 架构扩展灵活，易叠加分解、稀疏注意力与外生变量模块。

## 主要缺点
- 原生注意力内存/计算随序列长度二次增长。
- 小样本下参数冗余，泛化不稳定。
- 位置编码与尺度归一化不当会导致训练震荡。

## 针对缺点的解决办法
- 长序列优先使用 Informer/Autoformer/FEDformer 等高效变体。
- 采用patching、降采样、混合精度训练降低显存压力。
- 通过预训练时序表示或跨域迁移缓解数据不足问题。

## 常用参数
- `context_len`：历史窗口长度，决定可利用上下文范围。
- `d_model` 与 `n_heads`：控制表示能力与头间分工。
- `n_layers`：编码器深度，过深可能过拟合。
- `dropout`：抑制注意力层过拟合。
- `label_len/pred_len`（多步预测）：影响解码稳定性。

## 选型建议
- 当预测窗口长、变量多且需并行训练时，Transformer通常优于RNN。
- 若数据量有限且需快速上线，建议先用ARIMA/LSTM建立低成本基线。
- 部署端资源受限时优先选择轻量变体或蒸馏模型。

## 模型解释方法
- 分析多头注意力权重，识别关键历史时刻与关键变量通道。
- 对输入时间步做遮挡实验，度量各滞后区间对预测误差的边际影响。
- 在Autoformer类模型中分别解释趋势分量与季节分量贡献。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
