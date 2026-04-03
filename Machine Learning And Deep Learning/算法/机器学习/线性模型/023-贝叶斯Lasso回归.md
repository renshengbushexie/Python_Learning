# 023. 贝叶斯Lasso回归（Bayesian Lasso）

> 一级分类：机器学习
> 二级分类：线性模型
> 任务类型：回归
> 原始来源：readme_learning01.md / 监督学习

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| 贝叶斯Lasso回归 | 回归 | 基因筛选；文本稀疏回归；工业变量压缩 | 稀疏+区间输出；小样本稳；可层次扩展 | 推断慢；`lambda`敏感；实现复杂 | 预筛+EM；重采样定`lambda`；MCMC复核 | `alpha`;`fit_intercept`;`max_iter`;`tol` | 需稀疏且要不确定性时优先；仅点估计用Lasso；组效应转Elastic Net | 后验区间；零邻域概率；收敛诊断 |

## 适用场景
- **基因-表型回归**：需要稀疏变量集并报告不确定性区间。
- **文本情绪强度预测**：词项高维稀疏，贝叶斯框架可降低过拟合。
- **宏观因子精选**：从大量候选因子中提炼少量稳定驱动项。
- **工业机理变量筛选**：在可解释约束下保留关键传感器并量化置信度。
- **小样本科研建模**：样本少且特征多，需结合先验实现稳健估计。

## 模型实现原理
贝叶斯Lasso 在线性高斯似然下对每个系数施加 Laplace 先验：

$$
\mathbf y=X\mathbf w+\boldsymbol\varepsilon,
\quad \boldsymbol\varepsilon\sim\mathcal N(0,\sigma^2I),
$$

$$
p(w_j|\lambda)=\frac{\lambda}{2}\exp(-\lambda|w_j|),\quad j=1,\dots,p.
$$

后验众数等价于 Lasso：

$$
\hat{\mathbf w}_{MAP}=\arg\min_{\mathbf w}\frac{1}{2\sigma^2}\|\mathbf y-X\mathbf w\|_2^2+\lambda\|\mathbf w\|_1.
$$

Laplace 可写成尺度混合高斯（关键技巧）：

$$
w_j|\tau_j,\sigma^2\sim\mathcal N(0,\sigma^2\tau_j),
\quad
\tau_j\sim\mathrm{Exp}(\lambda^2/2).
$$

引入局部尺度变量 $\tau_j$ 后，条件后验为高斯，便于 EM/Gibbs 推断。

EM 思路：

- **E步**：计算 $\mathbb E[\tau_j^{-1}|w_j^{(t)},\sigma^{2(t)}]$；
- **M步**：解加权 Ridge：

$$
\mathbf w^{(t+1)}=\arg\min_{\mathbf w}
\frac{1}{2\sigma^2}\|\mathbf y-X\mathbf w\|_2^2+
\frac{1}{2}\sum_{j=1}^p\omega_j^{(t)}w_j^2,
$$

$$
\omega_j^{(t)}=\mathbb E[\tau_j^{-1}|\cdot],
\quad
\mathbf w^{(t+1)}=(X^\top X+\sigma^2\Omega^{(t)})^{-1}X^\top\mathbf y.
$$

关键性质：

1. **稀疏诱导**：Laplace 先验在 0 附近尖峰，推动系数向零集中。  
2. **概率化Lasso**：兼具 L1 稀疏性与后验不确定性输出。  
3. **层次模型可扩展**：可进一步加入组结构、异方差或层级先验。

## 主要优点
- **稀疏+不确定性并存**：得到关键特征同时给出参数区间。
- **小样本更稳**：先验抑制过拟合，提升泛化可靠性。
- **层次化灵活**：可扩展到组稀疏、异方差等复杂设定。
- **与Lasso兼容**：MAP解与经典Lasso一致，便于工程迁移。

## 主要缺点
- **计算成本高**：EM/MCMC 需要多轮矩阵运算，维度高时耗时增长明显。
- **超参数敏感**：$\lambda$ 过大导致过稀疏，过小则失去筛选效果。
- **实现门槛高**：需处理层次先验、数值稳定和收敛监控。
- **结果依赖推断方法**：变分近似可能低估后验方差。

## 针对缺点的解决办法
- **针对计算成本高**：  
  1) 先用 Lasso 预筛特征；2) 在候选子集上做贝叶斯推断。
- **针对超参数敏感**：  
  1) 对 $\lambda$ 做经验贝叶斯更新或CV校准；2) 用稳定性路径控制入模规模。
- **针对实现门槛高**：  
  1) 采用成熟概率编程框架（PyMC/Stan）；2) 设置收敛准则与多链诊断。
- **针对近似偏差**：  
  1) 用MCMC抽样对关键系数复核；2) 比较VI与MCMC的区间宽度差异。

## 常用参数
- `alpha: float`（MAP等价实现中对应L1强度，如 `sklearn.linear_model.Lasso`）。
- `fit_intercept: bool`，是否拟合截距。
- `max_iter: int`、`tol: float`，坐标下降或EM收敛控制。
- `selection: {'cyclic','random'}`，影响高维收敛效率。
- `warm_start: bool`，用于正则路径和增量拟合。

## 选型建议
- 需要“稀疏特征 + 可信区间”时，贝叶斯Lasso优于经典Lasso。
- 仅需要快速稀疏点估计，直接用 Lasso 成本更低。
- 相关特征成组保留需求强时，Elastic Net 通常更合适。
- 若强调变量相关性的完整后验结构，可考虑 Horseshoe 或 Spike-and-Slab。

## 模型解释方法
- **后验中位数/区间**：按可信区间是否跨0判断变量可靠性。
- **稀疏概率解释**：统计系数落在零邻域的后验质量评估“近零”概率。
- **路径与稳定性**：比较不同 $\lambda$ 下入模集合变化。
- **收敛诊断**：监控EM目标或MCMC链的R-hat、有效样本数。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
