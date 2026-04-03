# 036. Apriori关联规则（Apriori）

> 一级分类：机器学习
> 二级分类：推荐与交互建模
> 任务类型：关联规则
> 原始来源：readme_learning01.md / 无监督学习

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| Apriori关联规则 | 关联规则 | 购物篮分析、交叉销售规则挖掘 | 规则可解释性强，业务可直接落地 | 候选项集数量随维度指数增长 | 提高最小支持度+剪枝+按品类分桶 | minsup、minconf、minlift、最大项集长度 | 数据中等规模且重解释时优先 | 用支持度/置信度/提升度解释规则质量 |

## 适用场景
- 商超购物篮共购分析。
- 电商“买了A还买B”规则发现。
- 促销组合与捆绑套餐设计。

## 模型实现原理
事务集合记为 $T$。项集 $A$ 的支持度
$$
\mathrm{sup}(A)=\frac{|\{t\in T\mid A\subseteq t\}|}{|T|}.
$$
规则 $A\Rightarrow B$ 的置信度与提升度：
$$
\mathrm{conf}(A\Rightarrow B)=\frac{\mathrm{sup}(A\cup B)}{\mathrm{sup}(A)},\qquad
\mathrm{lift}(A\Rightarrow B)=\frac{\mathrm{conf}(A\Rightarrow B)}{\mathrm{sup}(B)}.
$$
Apriori 利用反单调性：若 $A$ 非频繁，则任何超集 $A\cup C$ 均非频繁。算法按 $k=1,2,\dots$ 逐层生成候选 $C_k$，通过数据库扫描筛出频繁集 $L_k$。最坏情形复杂度接近指数级，瓶颈在候选生成与多轮扫描。

## 主要优点
- 规则语义清晰，业务人员可直接验证与执行。
- 不依赖复杂模型训练，部署简单。
- 适合探索式分析与策略启发。

## 主要缺点
- 高频长尾并存时候选集会爆炸。
- 需要多次全库扫描，I/O 成本高。
- 对稀疏超高维数据效率不佳。

## 针对缺点的解决办法
- 先按业务最小交易频次过滤低价值商品。
- 分品类、分时段建模降低候选空间。
- 大规模场景改用 FP-Growth 或并行化实现。

## 常用参数
- minsup：频繁项集阈值。
- minconf：规则置信度阈值。
- minlift：关联强度阈值（常设 >1）。
- max_len：限制项集最大长度。

## 选型建议
- 规则解释优先、数据规模中小：选 Apriori。
- 若事务量大且维度高：优先 FP-Growth。
- 线上策略应先做离线回放验证收益。

## 模型解释方法
- 对每条规则输出 $(\mathrm{sup},\mathrm{conf},\mathrm{lift})$ 三元组。
- 结合时间窗口监控规则稳定性与季节性漂移。
- 通过 A/B 测试验证规则触发后的实际转化提升。

## 参考资料
- [Legacy Source: readme_learning01.md](../../../归档/readme_learning01.md)
- [Scikit-learn 文档](https://scikit-learn.org/stable/)
