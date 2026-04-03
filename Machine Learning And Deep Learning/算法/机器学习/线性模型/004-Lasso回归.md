# 004. Lasso回归（Lasso Regression）

> 一级分类：机器学习
> 二级分类：线性模型
> 任务类型：回归
> 原始来源：readme_learning01.md / 监督学习

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| Lasso回归 | 回归 | 文本稀疏特征；基因筛选；风控压缩 | 稀疏解；自动选特征；降方差 | 相关特征择一；收缩偏差；路径不稳 | Elastic Net；重采样稳定选择；Post-Lasso | `alpha`;`max_iter`;`tol`;`selection` | 真解稀疏优先；相关强转Elastic Net；仅稳预测用Ridge | 非零特征集；正则路径；KKT诊断 |

## 适用场景
- **广告点击率回归**：One-hot 特征海量且稀疏，Lasso可直接做自动筛选。
- **基因组学连续表型**：候选基因远多于样本，L1约束提升可解释性。
- **信用定价模型**：大量行为变量中仅少数有效，稀疏解便于监管审计。
- **文本情感强度预测**：词袋特征高维稀疏，Lasso可去除弱词项。
- **工业变量压缩**：传感器冗余时可压缩输入维度，降低部署成本。

## 模型实现原理
Lasso 在平方损失上加入 $L_1$ 正则：

$$
\hat y_i=\mathbf w^\top\mathbf x_i+b,
\quad
J(\mathbf w,b)=\frac{1}{2n}\sum_{i=1}^n(y_i-\mathbf w^\top\mathbf x_i-b)^2+\alpha\|\mathbf w\|_1.
$$

其中 $\|\mathbf w\|_1=\sum_{j=1}^p|w_j|$。因 $|w_j|$ 在 0 点不可导，使用次梯度：

$$
\partial |w_j|=
\begin{cases}
\{+1\}, & w_j>0\\
[-1,1], & w_j=0\\
\{-1\}, & w_j<0
\end{cases}
$$

KKT 条件（中心化后）为

$$
\frac{1}{n}\mathbf x_j^\top(\mathbf y-X\mathbf w)-\alpha s_j=0,
\quad s_j\in\partial|w_j|.
$$

由此可得稀疏判据：若
$\left|\frac{1}{n}\mathbf x_j^\top(\mathbf r_{-j})\right|\le\alpha$，则 $w_j^*=0$。

坐标下降更新（标准化特征）为软阈值：

$$
\rho_j=\frac{1}{n}\mathbf x_j^\top\Big(\mathbf y-\sum_{k\ne j}\mathbf x_kw_k\Big),
\quad
w_j\leftarrow S(\rho_j,\alpha),
$$

$$
S(z,\alpha)=\mathrm{sign}(z)\max(|z|-\alpha,0).
$$

关键性质：

1. **凸但非光滑**：全局最优存在，可用坐标下降高效求解。  
2. **几何稀疏性**：$L_1$ 约束区域有尖角，最优点更易落在坐标轴上。  
3. **变量选择一致性有条件**：受 irrepresentable condition 与样本规模影响。

## 主要优点
- **天然稀疏建模**：无效特征系数直接为零，推理链路更短。
- **高维适配好**：在 $p\gg n$ 场景仍可得到可用解。
- **解释简洁**：保留特征数量可控，报告与规则提炼成本低。
- **工程节省资源**：减少特征存储、计算和监控指标数量。

## 主要缺点
- **相关组随机选择**：强相关变量中常只留一个，业务解释可能片面。
- **收缩偏差**：即使入模，大系数也被系统性低估。
- **解路径不稳定**：小样本高噪声下，入模集合对采样扰动敏感。
- **对特征缩放敏感**：未标准化时惩罚不公平，选择结果失真。

## 针对缺点的解决办法
- **针对相关组随机选择**：  
  1) 改用 `ElasticNet(l1_ratio<1)` 引入组效应；2) 先做相关性聚类后组内代表筛选。
- **针对收缩偏差**：  
  1) 执行 Post-Lasso：在非零特征上再做 OLS/Ridge；2) 报告偏差校正区间。
- **针对路径不稳定**：  
  1) Bootstrap 多次拟合统计入选频率；2) 仅保留频率超过阈值的稳定变量。
- **针对缩放敏感**：  
  1) 统一标准化；2) 对类别变量采用频率编码/目标编码后再正则化。

## 常用参数
- `alpha: float`，L1强度；通常对数网格 $10^{-4}\sim10^1$。
- `fit_intercept: bool`，是否拟合截距。
- `max_iter: int`、`tol: float`，控制坐标下降收敛。
- `selection: {'cyclic','random'}`，随机坐标在高维时常更快。
- `warm_start: bool`，用于正则路径连续训练。

## 选型建议
- 若预期有效特征占比低（如 <20%），Lasso通常优于 Ridge。
- 若特征高度相关且需“成组保留”，优先 Elastic Net。
- 若目标仅是最低预测误差而非稀疏解释，可比较 Ridge/GBDT。
- 在样本极少时，先做降维或先验筛选再用Lasso更稳。

## 模型解释方法
- **非零系数集合解释**：入模变量即被算法选择的关键特征。
- **正则路径解释**：观察 `alpha` 下降时变量入模顺序，评估稳健性。
- **KKT 诊断**：检查零系数对应相关性是否落在阈值区间内。
- **稳定性报告**：Bootstrap 入选频率用于衡量可重复性。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
- [Lasso回归特征选择示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html)
