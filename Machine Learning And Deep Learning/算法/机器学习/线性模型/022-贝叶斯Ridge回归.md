# 022. 贝叶斯Ridge回归（Bayesian Ridge Regression）

> 一级分类：机器学习
> 二级分类：线性模型
> 任务类型：回归
> 原始来源：readme_learning01.md / 监督学习

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| 贝叶斯Ridge回归 | 回归 | 小样本实验；剂量反应；工业标定 | 后验分布；自动估计精度；抗过拟合 | 计算重；先验敏感；非稀疏 | PCA降维；调`alpha_1/2`,`lambda_1/2`；转ARD | `alpha_1`;`alpha_2`;`lambda_1`;`lambda_2` | 需区间预测优先；只要点预测用Ridge；稀疏需求转ARD/Lasso | 后验均值方差；区间覆盖率；证据轨迹 |

## 适用场景
- **药效实验回归**：样本少、噪声大，需要预测区间支持试验决策。
- **工程传感器标定**：需给出点估计与不确定性，便于容差设计。
- **材料参数反演**：反演变量相关且样本昂贵，贝叶斯正则更稳。
- **金融风险敞口估计**：要求区间风险而非单点值。
- **A/B小流量评估**：低样本阶段需避免过拟合并量化可信度。

## 模型实现原理
贝叶斯Ridge将线性回归写成概率图模型：

$$
\mathbf y=X\mathbf w+\boldsymbol\varepsilon,
\quad \boldsymbol\varepsilon\sim\mathcal N(\mathbf 0,\beta^{-1}I).
$$

先验（权重高斯）：

$$
p(\mathbf w|\alpha)=\mathcal N(\mathbf 0,\alpha^{-1}I).
$$

似然：

$$
p(\mathbf y|\mathbf w,\beta)=\mathcal N(X\mathbf w,\beta^{-1}I).
$$

由贝叶斯公式得后验：

$$
p(\mathbf w|\mathbf y,\alpha,\beta)
\propto p(\mathbf y|\mathbf w,\beta)p(\mathbf w|\alpha)
=\mathcal N(\mathbf m_N,S_N),
$$

$$
S_N=(\alpha I+\beta X^\top X)^{-1},
\quad
\mathbf m_N=\beta S_NX^\top\mathbf y.
$$

这与 Ridge 解一致于后验均值/众数。

预测分布（新样本 $\mathbf x_*$）：

$$
p(y_*|\mathbf x_*,\mathcal D)=\mathcal N\big(\mathbf x_*^\top\mathbf m_N,
\beta^{-1}+\mathbf x_*^\top S_N\mathbf x_*\big).
$$

超参数通常用证据最大化（Type-II ML）迭代估计：

$$
\gamma=\sum_{j=1}^p\frac{\beta\lambda_j}{\alpha+\beta\lambda_j},
\quad
\alpha\leftarrow\frac{\gamma}{\|\mathbf m_N\|_2^2},
\quad
\beta\leftarrow\frac{n-\gamma}{\|\mathbf y-X\mathbf m_N\|_2^2},
$$

其中 $\lambda_j$ 为 $X^\top X$ 的特征值。

关键性质：

1. **后验闭式**：共轭先验下可得解析高斯后验。  
2. **不确定性显式量化**：预测方差分为噪声项与参数不确定性项。  
3. **岭回归概率解释**：MAP 与 Ridge 目标等价。

## 主要优点
- **区间预测可用**：直接给出均值与方差，支持风险控制。
- **超参数自适应**：证据最大化可自动平衡拟合与收缩。
- **小样本稳健**：先验约束降低参数波动。
- **可解释统计完整**：可报告后验均值、方差、可信区间。

## 主要缺点
- **计算复杂度较高**：涉及 $p\times p$ 矩阵分解，时间约 $O(p^3)$。
- **超先验影响明显**：`alpha_1/2`,`lambda_1/2` 会改变收缩力度与区间宽度。
- **稀疏性不足**：参数通常连续收缩不为零，变量选择能力有限。

## 针对缺点的解决办法
- **针对计算复杂度**：  
  1) 先做 PCA/特征聚合降维；2) 使用数值稳定的 Cholesky/SVD 实现。
- **针对超先验敏感**：  
  1) 以对数网格扫描 `alpha_1,alpha_2,lambda_1,lambda_2`；2) 观察预测区间覆盖率校准。
- **针对稀疏性不足**：  
  1) 变量筛选后再贝叶斯Ridge；2) 需要自动稀疏时改用 ARDRegression 或 Bayesian Lasso。

## 常用参数
- `max_iter: int`，迭代上限（证据最大化过程）。
- `tol: float`，收敛阈值。
- `alpha_1, alpha_2: float`，噪声精度先验 Gamma 分布参数。
- `lambda_1, lambda_2: float`，权重精度先验 Gamma 参数。
- `fit_intercept: bool`、`compute_score: bool`（是否记录边际似然轨迹）。

## 选型建议
- 当样本较少且必须提供预测不确定性时，优先贝叶斯Ridge。
- 若只关心点预测且追求训练速度，普通 Ridge 常更轻量。
- 若需求是自动稀疏特征选择，优先 ARDRegression 或 Lasso 系列。
- 关系明显非线性时，考虑核岭回归或树模型并配区间估计。

## 模型解释方法
- **后验参数解释**：报告 $\mathbf m_N$ 与对角方差 $\mathrm{diag}(S_N)$。
- **不确定性分解**：区分观测噪声 $\beta^{-1}$ 与参数不确定性 $\mathbf x_*^\top S_N\mathbf x_*$。
- **覆盖率评估**：检查 95% 预测区间经验覆盖率是否接近 95%。
- **证据轨迹**：观察边际似然随迭代变化，判断超参数是否收敛。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
- [贝叶斯Ridge教程](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression)
