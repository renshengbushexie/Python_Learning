# 273. 蒙特卡洛Dropout（Monte Carlo Dropout, MC Dropout）

> 一级分类：机器学习
> 二级分类：概率统计
> 任务类型：不确定性建模
> 原始来源：readme_learning02.md / 概率推断与不确定性

## 算法速查表
| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |
|---|---|---|---|---|---|---|---|---|
| 蒙特卡洛Dropout | 不确定性建模 | 风险敏感预测；主动学习样本选择；OOD检测基线 | 复用现有Dropout网络即可估计不确定性；实现成本低 | 属于近似变分推断；方差估计受drop率与采样数影响 | 调整drop率与前向采样次数$T$；做温度缩放改善校准 | dropout rate、采样次数$T$、模型精度参数$\tau$、温度参数 | 需快速接入不确定性时优先；高精度贝叶斯后验需更强方法 | 查看预测均值/方差、熵与互信息曲线 |

## 适用场景
- 已有Dropout模型但需输出置信区间的线上系统。
- 需要按不确定性选样本标注的主动学习流程。
- 医疗/工业告警中需要“低置信度拒判”的分类任务。

## 模型实现原理
Gal & Ghahramani将Dropout解释为近似变分推断。对权重$W$设变分分布
$$
q_\theta(W)=\prod_l q(W_l),
$$
其中每层通过Bernoulli掩码实现：
$$
\tilde{W}_l=M_l\odot W_l,\quad (M_l)_{ij}\sim\mathrm{Bernoulli}(p_l).
$$
训练时最小化目标可写为
$$
\mathcal{L}=\frac{1}{N}\sum_{n=1}^N\ell\big(y_n,f(x_n;\tilde{W})\big)+\lambda\sum_l\|W_l\|_2^2,
$$
对应于变分自由能近似。

测试时保持Dropout开启并做$T$次随机前向：
$$
\hat{y}^{(t)}=f(x;\tilde{W}^{(t)}),\quad t=1,\dots,T.
$$
预测均值
$$
\bar{y}=\frac{1}{T}\sum_{t=1}^T\hat{y}^{(t)}.
$$
回归总不确定性近似
$$
\mathrm{Var}[y\mid x]\approx \underbrace{\frac{1}{T}\sum_{t=1}^{T}(\hat{y}^{(t)}-\bar{y})^2}_{\text{epistemic}}+\underbrace{\tau^{-1}}_{\text{aleatoric baseline}}.
$$
分类中可用预测熵
$$
H(\bar{p})=-\sum_c\bar{p}_c\log\bar{p}_c,\quad \bar{p}=\frac1T\sum_t p^{(t)}.
$$

## 主要优点
- 不改模型结构即可获得后验采样近似，迁移成本低。
- 相比深度集成，参数量不增加，仅增加测试前向次数。
- 可直接输出逐样本不确定性用于拒判和风险控制。

## 主要缺点
- 不确定性估计高度依赖dropout放置位置与概率设定。
- 对强分布外样本常出现过度自信，需额外校准。
- 采样次数$T$过小方差噪声大，过大则推理延迟升高。

## 针对缺点的解决办法
- 在验证集上联合搜索`dropout rate`与$T$，以NLL/ECE为准。
- 叠加温度缩放、深度集成或SWAG提升校准质量。
- 对关键层（如最后几层）启用MC Dropout，平衡成本与收益。

## 常用参数
- `dropout_p`：每层丢弃概率。
- `mc_samples(T)`：推理采样次数。
- `tau`：观测噪声精度（回归）。
- `temperature`：分类后处理校准参数。

## 选型建议
- 预算有限且已有Dropout网络时，MC Dropout是最快的不确定性升级路径。
- 若任务强依赖OOD鲁棒性，优先深度集成或贝叶斯最后层方法并与MC对比。

## 模型解释方法
- 绘制样本级方差与误差散点图，验证“不确定性-错误率”一致性。
- 统计高熵样本类别分布，定位数据集边界与标注噪声区域。
- 监控ECE/NLL随$T$变化，选择收益递减拐点。

## 参考资料
- [Legacy Source: readme_learning02.md](../../../归档/readme_learning02.md)
