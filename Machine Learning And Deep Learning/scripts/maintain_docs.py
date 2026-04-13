from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = PROJECT_ROOT / "算法"
ROOT_README = PROJECT_ROOT / "README.md"
ROOT_README_CN = PROJECT_ROOT / "README_CN.md"
TAXONOMY = PROJECT_ROOT / "taxonomy.md"

PRIMARY_ORDER = ["机器学习", "深度学习"]
SECONDARY_DISPLAY_OVERRIDES = {
    ("机器学习", "距离与邻近法"): "距离/邻近法",
    ("深度学习", "自动机器学习与神经架构搜索"): "AutoML与神经架构搜索",
}
SECONDARY_CANONICAL_NAMES = {
    "距离/邻近法": "距离与邻近法",
    "距离与邻近法": "距离与邻近法",
    "AutoML与神经架构搜索": "自动机器学习与神经架构搜索",
    "自动机器学习与神经架构搜索": "自动机器学习与神经架构搜索",
}
LOCALIZED_FILENAME_TITLES = {
    4: "套索回归",
    22: "贝叶斯岭回归",
    23: "贝叶斯套索回归",
    24: "均值聚类",
    32: "学生分布随机近邻嵌入",
    36: "先验关联规则",
    37: "频繁模式增长挖掘",
    42: "深度价值网络",
    44: "行动者评论家方法",
    63: "U型网络",
    75: "随机失活正则化",
    86: "信息增益决策树",
    87: "增益率决策树",
    88: "分类回归决策树",
    98: "双向编码器表征预训练模型",
    102: "季节性自回归积分滑动平均模型",
    103: "先知时间序列模型",
    104: "长短期记忆网络时间序列模型",
    105: "变换器时间序列模型",
    107: "一类支持向量机",
    109: "椭圆包络",
    110: "密度聚类异常检测",
    112: "倾向得分匹配",
    115: "因果森林",
    117: "原型网络",
    118: "记忆增强网络",
    119: "学会学习",
    121: "联邦近端优化",
    127: "沙普利加性解释",
    128: "集成梯度",
    129: "注意力可视化",
    130: "反事实解释",
    131: "量子神经网络",
    132: "变分量子算法",
    133: "量子支持向量机",
    136: "差分进化",
    137: "演化策略",
    138: "风格生成对抗网络",
    139: "扩散模型",
    141: "图像到图像翻译",
    145: "知识蒸馏",
    146: "模型剪枝",
    147: "模型量化",
    148: "低秩近似",
    149: "移动网络",
    150: "神经常微分方程",
    157: "感知机在线算法",
    158: "正则化领袖跟随",
    170: "深度上下文化词表示",
    175: "哈尔特征与级联分类器",
    176: "词频逆文档频率",
    177: "潜在语义分析",
    178: "潜在狄利克雷分配",
    183: "连接时序分类语音识别",
    184: "听觉注意拼写模型",
    196: "自适应矩估计优化器",
    197: "自适应梯度",
    198: "均方根传播",
    201: "参数服务器",
    202: "全归约",
    204: "异步随机梯度下降",
    212: "快速梯度符号法",
    213: "投影梯度下降",
    225: "时空变换器",
    255: "层次视觉变换器",
    273: "蒙特卡洛随机失活",
    301: "混合专家变换器架构",
    302: "递归神经变换器",
    303: "自适应计算变换器",
    346: "自我进化人工智能系统",
    348: "意识级人工智能架构",
    350: "元宇宙人工智能生态",
    351: "自适应词元化变换器",
    352: "分层记忆变换器",
    356: "脱氧核糖核酸序列深度学习",
    358: "合成生物学人工智能",
    365: "微服务人工智能架构",
    374: "文化感知人工智能",
    375: "道德推理人工智能",
    377: "循环人工智能架构",
    378: "环境监测人工智能",
    380: "可再生能源人工智能",
    381: "增强现实人工智能",
    385: "现实感知人工智能",
    390: "技能传递人工智能",
    392: "实验设计人工智能",
    395: "科学创新人工智能",
    399: "自我复制人工智能系统",
    400: "奇点级人工智能架构",
}
# Derived from the implementation plan plus the audit-detected placeholder / truncated names.
FORCED_RENAME_IDS = {
    63,
    66,
    67,
    86,
    87,
    88,
    98,
    103,
    107,
    109,
    112,
    115,
    117,
    118,
    119,
    121,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    136,
    137,
    138,
    139,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    158,
    170,
    176,
    177,
    178,
    184,
    197,
    198,
    201,
    202,
    212,
    213,
    222,
    252,
}

FRONT_MATTER_PATTERNS = {
    "primary": re.compile(r"^> 一级分类：(.+)$", re.MULTILINE),
    "secondary": re.compile(r"^> 二级分类：(.+)$", re.MULTILINE),
    "task_type": re.compile(r"^> 任务类型：(.+)$", re.MULTILINE),
}
HEADING_PATTERN = re.compile(r"^#\s*(\d{3})\.\s*(.+?)\s*$")
TITLE_SPLIT_PATTERN = re.compile(r"^(?P<main>.+?)\s*[（(](?P<alt>.+?)[)）]\s*$")
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REFERENCE_SECTION_PATTERN = re.compile(r"^## 参考资料[\s\S]*$", re.MULTILINE)

CATEGORY_REFERENCE_GUIDANCE: dict[tuple[str, str], list[str]] = {
    ("机器学习", "专用领域算法"): [
        "Murphy, K. P. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.",
        "周志华. 《机器学习》. 清华大学出版社, 2016.",
    ],
    ("机器学习", "元学习"): [
        "Finn, C., Abbeel, P., & Levine, S. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML, 2017.",
        "Hospedales, T., et al. Meta-Learning in Neural Networks: A Survey. IEEE TPAMI, 2022.",
    ],
    ("机器学习", "其他技术"): [
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.",
        "Murphy, K. P. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.",
    ],
    ("机器学习", "分布式机器学习"): [
        "Dean, J., et al. Large Scale Distributed Deep Networks. NeurIPS, 2012.",
        "Li, M., et al. Scaling Distributed Machine Learning with the Parameter Server. OSDI, 2014.",
    ],
    ("机器学习", "可解释AI"): [
        "Ribeiro, M. T., Singh, S., & Guestrin, C. Why Should I Trust You? Explaining the Predictions of Any Classifier. KDD, 2016.",
        "Lundberg, S. M., & Lee, S.-I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.",
    ],
    ("机器学习", "因果推断"): [
        "Pearl, J. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2009.",
        "Imbens, G. W., & Rubin, D. B. Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press, 2015.",
    ],
    ("机器学习", "在线学习"): [
        "Shalev-Shwartz, S. Online Learning and Online Convex Optimization. Foundations and Trends in Machine Learning, 2012.",
        "Cesa-Bianchi, N., & Lugosi, G. Prediction, Learning, and Games. Cambridge University Press, 2006.",
    ],
    ("机器学习", "异常检测"): [
        "Chandola, V., Banerjee, A., & Kumar, V. Anomaly Detection: A Survey. ACM Computing Surveys, 2009.",
        "Aggarwal, C. C. Outlier Analysis. Springer, 2017.",
    ],
    ("机器学习", "强化学习"): [
        "Sutton, R. S., & Barto, A. G. Reinforcement Learning: An Introduction. MIT Press, 2018.",
        "Mnih, V., et al. Human-level Control through Deep Reinforcement Learning. Nature, 2015.",
    ],
    ("机器学习", "推荐与交互建模"): [
        "Koren, Y., Bell, R., & Volinsky, C. Matrix Factorization Techniques for Recommender Systems. IEEE Computer, 2009.",
        "Ricci, F., Rokach, L., & Shapira, B. Recommender Systems Handbook. Springer, 2022.",
    ],
    ("机器学习", "时间序列"): [
        "Hamilton, J. D. Time Series Analysis. Princeton University Press, 1994.",
        "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. Time Series Analysis: Forecasting and Control. Wiley, 2015.",
    ],
    ("机器学习", "树模型"): [
        "Breiman, L., Friedman, J., Olshen, R., & Stone, C. Classification and Regression Trees. Wadsworth, 1984.",
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 9.",
        "周志华. 《机器学习》. 清华大学出版社, 2016. 第4章.",
    ],
    ("机器学习", "核方法"): [
        "Vapnik, V. The Nature of Statistical Learning Theory. Springer, 1995.",
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 6-7.",
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 12.",
    ],
    ("机器学习", "概率统计"): [
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 2, 8, 9, 13.",
        "Murphy, K. P. Machine Learning: A Probabilistic Perspective. MIT Press, 2012. Chapter 10-11.",
    ],
    ("机器学习", "正则化与优化"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 11.",
        "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016. Chapter 7-8.",
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Section 3.4.",
    ],
    ("机器学习", "线性模型"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 3.",
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 3-4.",
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 3-4.",
        "周志华. 《机器学习》. 清华大学出版社, 2016. 第3章.",
    ],
    ("机器学习", "联邦学习"): [
        "McMahan, H. B., et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, 2017.",
        "Kairouz, P., et al. Advances and Open Problems in Federated Learning. Foundations and Trends in Machine Learning, 2021.",
    ],
    ("机器学习", "聚类"): [
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 9.",
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 14.",
        "周志华. 《机器学习》. 清华大学出版社, 2016. 第9章.",
    ],
    ("机器学习", "距离与邻近法"): [
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 13.",
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 2.",
    ],
    ("机器学习", "进化计算"): [
        "Back, T., Fogel, D. B., & Michalewicz, Z. Handbook of Evolutionary Computation. Oxford University Press, 1997.",
        "Eiben, A. E., & Smith, J. E. Introduction to Evolutionary Computing. Springer, 2015.",
    ],
    ("机器学习", "量子机器学习"): [
        "Schuld, M., & Petruccione, F. Supervised Learning with Quantum Computers. Springer, 2019.",
        "Biamonte, J., et al. Quantum Machine Learning. Nature, 2017.",
    ],
    ("机器学习", "降维"): [
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 14.",
        "Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006. Chapter 12.",
    ],
    ("机器学习", "集成学习"): [
        "Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning. Springer, 2009. Chapter 8, 10, 16.",
        "周志华. 《机器学习》. 清华大学出版社, 2016. 第8章.",
    ],
    ("深度学习", "自动机器学习与神经架构搜索"): [
        "Zoph, B., & Le, Q. V. Neural Architecture Search with Reinforcement Learning. ICLR, 2017.",
        "Liu, H., Simonyan, K., & Yang, Y. DARTS: Differentiable Architecture Search. ICLR, 2019.",
    ],
    ("深度学习", "前沿研究方向"): [
        "LeCun, Y., Bengio, Y., & Hinton, G. Deep Learning. Nature, 2015.",
        "Bommasani, R., et al. On the Opportunities and Risks of Foundation Models. arXiv, 2021.",
    ],
    ("深度学习", "前馈神经网络"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 4-5.",
        "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016. Chapter 6.",
    ],
    ("深度学习", "可信与鲁棒"): [
        "Goodfellow, I. J., Shlens, J., & Szegedy, C. Explaining and Harnessing Adversarial Examples. ICLR, 2015.",
        "Madry, A., et al. Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR, 2018.",
    ],
    ("深度学习", "基础模型与大模型"): [
        "Bommasani, R., et al. On the Opportunities and Risks of Foundation Models. arXiv, 2021.",
        "Brown, T., et al. Language Models are Few-Shot Learners. NeurIPS, 2020.",
        "Touvron, H., et al. LLaMA: Open and Efficient Foundation Language Models. arXiv, 2023.",
    ],
    ("深度学习", "多模态学习"): [
        "Radford, A., et al. Learning Transferable Visual Models from Natural Language Supervision. ICML, 2021.",
        "Baltrusaitis, T., Ahuja, C., & Morency, L.-P. Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI, 2019.",
    ],
    ("深度学习", "循环与序列网络"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 9-10.",
        "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016. Chapter 10.",
    ],
    ("深度学习", "时空建模"): [
        "Shi, X., et al. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. NeurIPS, 2015.",
        "Wu, Z., et al. A Comprehensive Survey on Graph Neural Networks. IEEE TNNLS, 2021.",
    ],
    ("深度学习", "正则化与优化"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 11.",
        "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016. Chapter 7-8.",
    ],
    ("深度学习", "生成模型"): [
        "Kingma, D. P., & Welling, M. Auto-Encoding Variational Bayes. ICLR, 2014.",
        "Goodfellow, I. J., et al. Generative Adversarial Nets. NeurIPS, 2014.",
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 17.",
    ],
    ("深度学习", "神经渲染与合成"): [
        "Mildenhall, B., et al. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV, 2020.",
        "Tewari, A., et al. Advances in Neural Rendering. Computer Graphics Forum, 2022.",
    ],
    ("深度学习", "神经符号"): [
        "Garcez, A. d'A., Lamb, L. C., & Gabbay, D. Neural-Symbolic Cognitive Reasoning. Springer, 2009.",
        "Garcez, A. d'A., et al. Neural-Symbolic Learning and Reasoning: Contributions and Challenges. AAAI Spring Symposium, 2019.",
    ],
    ("深度学习", "科学计算与物理建模"): [
        "Raissi, M., Perdikaris, P., & Karniadakis, G. E. Physics-informed Neural Networks. Journal of Computational Physics, 2019.",
        "Karniadakis, G. E., et al. Physics-informed Machine Learning. Nature Reviews Physics, 2021.",
    ],
    ("深度学习", "程序合成与神经编译"): [
        "Gulwani, S., Polozov, O., & Singh, R. Program Synthesis. Foundations and Trends in Programming Languages, 2017.",
        "Parisotto, E., et al. Neural Programmer-Interpreters. ICLR, 2016.",
    ],
    ("深度学习", "自然语言处理"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 14-15.",
        "Jurafsky, D., & Martin, J. H. Speech and Language Processing. Pearson, 2023.",
    ],
    ("深度学习", "自监督学习"): [
        "Chen, T., et al. A Simple Framework for Contrastive Learning of Visual Representations. ICML, 2020.",
        "He, K., et al. Masked Autoencoders Are Scalable Vision Learners. CVPR, 2022.",
    ],
    ("深度学习", "表示学习"): [
        "Bengio, Y., Courville, A., & Vincent, P. Representation Learning: A Review and New Perspectives. IEEE TPAMI, 2013.",
        "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016. Chapter 14-15.",
    ],
    ("深度学习", "视觉模型"): [
        "Zhang, A., Lipton, Z. C., Li, M., & Smola, A. Dive into Deep Learning. Cambridge University Press, 2023. Chapter 7-8, 13.",
        "Krizhevsky, A., Sutskever, I., & Hinton, G. E. ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS, 2012.",
        "He, K., et al. Deep Residual Learning for Image Recognition. CVPR, 2016.",
    ],
    ("深度学习", "认知与记忆"): [
        "Graves, A., Wayne, G., & Danihelka, I. Neural Turing Machines. arXiv, 2014.",
        "Graves, A., et al. Hybrid Computing Using a Neural Network with Dynamic External Memory. Nature, 2016.",
    ],
    ("深度学习", "语音处理"): [
        "Jurafsky, D., & Martin, J. H. Speech and Language Processing. Pearson, 2023.",
        "Baevski, A., et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS, 2020.",
    ],
    ("深度学习", "边缘与高效推理"): [
        "Han, S., Mao, H., & Dally, W. J. Deep Compression. ICLR, 2016.",
        "Howard, A. G., et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv, 2017.",
    ],
    ("深度学习", "迁移与持续学习"): [
        "Pan, S. J., & Yang, Q. A Survey on Transfer Learning. IEEE TKDE, 2010.",
        "De Lange, M., et al. A Continual Learning Survey. IEEE TPAMI, 2022.",
    ],
}


@dataclass(frozen=True)
class AlgorithmDoc:
    algo_id: int
    main_title: str
    alt_title: str | None
    primary: str
    secondary: str
    task_type: str
    path: Path
    relative_path: Path
    text: str

    @property
    def current_filename(self) -> str:
        return self.path.name

    @property
    def current_stem(self) -> str:
        return self.path.stem

    @property
    def link_title(self) -> str:
        return f"{self.algo_id:03d}. {localized_title_for_doc(self)}"

    @property
    def canonical_filename(self) -> str:
        return f"{self.algo_id:03d}-{slugify_title(localized_title_for_doc(self))}.md"

    @property
    def taxonomy_link(self) -> str:
        return self.relative_path.as_posix()


def slugify_title(title: str) -> str:
    title = title.replace("&", "and")
    buf: list[str] = []
    for char in title:
        if char.isalnum() or char in {"-", ".", "+"}:
            buf.append(char)
        elif char.isspace():
            buf.append("-")
        else:
            buf.append("-")
    slug = re.sub(r"-{2,}", "-", "".join(buf)).strip("-.")
    if not slug:
        raise ValueError(f"Could not derive canonical filename from title: {title!r}")
    return slug


def split_title(title: str) -> tuple[str, str | None]:
    match = TITLE_SPLIT_PATTERN.match(title.strip())
    if match:
        return match.group("main").strip(), match.group("alt").strip()
    return title.strip(), None


def normalize_secondary_name(name: str) -> str:
    return SECONDARY_CANONICAL_NAMES.get(name, name)


def localized_title_for_doc(doc: AlgorithmDoc) -> str:
    return LOCALIZED_FILENAME_TITLES.get(doc.algo_id, doc.main_title)


def extract_title_segment(stem: str) -> str:
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 else stem


def requires_canonical_rename(doc: AlgorithmDoc) -> bool:
    current_segment = extract_title_segment(doc.current_stem)
    return (
        doc.algo_id in LOCALIZED_FILENAME_TITLES
        or doc.algo_id in FORCED_RENAME_IDS
        or bool(re.fullmatch(r"算法\d{3}", current_segment))
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def parse_doc(path: Path) -> AlgorithmDoc:
    text = read_text(path)
    first_line = text.splitlines()[0].strip()
    heading_match = HEADING_PATTERN.match(first_line)
    if not heading_match:
        raise ValueError(f"Unsupported heading format: {path}")

    algo_id = int(heading_match.group(1))
    raw_title = heading_match.group(2).strip()
    main_title, alt_title = split_title(raw_title)

    metadata: dict[str, str] = {}
    for field, pattern in FRONT_MATTER_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Missing {field} metadata: {path}")
        metadata[field] = match.group(1).strip()

    return AlgorithmDoc(
        algo_id=algo_id,
        main_title=main_title,
        alt_title=alt_title,
        primary=metadata["primary"],
        secondary=normalize_secondary_name(metadata["secondary"]),
        task_type=metadata["task_type"],
        path=path,
        relative_path=path.relative_to(PROJECT_ROOT),
        text=text,
    )


def iter_algorithm_files() -> Iterable[Path]:
    for path in sorted(ALGORITHM_ROOT.rglob("*.md")):
        if path.name == "README.md":
            continue
        yield path


def load_docs() -> list[AlgorithmDoc]:
    docs = [parse_doc(path) for path in iter_algorithm_files()]
    docs.sort(key=lambda doc: doc.algo_id)
    return docs


def build_category_map(docs: Iterable[AlgorithmDoc]) -> dict[str, dict[str, list[AlgorithmDoc]]]:
    data: dict[str, dict[str, list[AlgorithmDoc]]] = {primary: {} for primary in PRIMARY_ORDER}
    for doc in docs:
        data.setdefault(doc.primary, {})
        data[doc.primary].setdefault(doc.secondary, []).append(doc)
    for primary_groups in data.values():
        for entries in primary_groups.values():
            entries.sort(key=lambda item: item.algo_id)
    return data


def render_primary_readme(primary: str, grouped_docs: dict[str, list[AlgorithmDoc]]) -> str:
    total = sum(len(entries) for entries in grouped_docs.values())
    lines = [
        f"# {primary}",
        "",
        f"> 算法数量：{total}",
        "",
        "## 二级分类导航",
        "| 二级分类 | 算法数量 | 链接 |",
        "|---|---:|---|",
    ]
    for secondary in sorted(grouped_docs):
        display = SECONDARY_DISPLAY_OVERRIDES.get((primary, secondary), secondary)
        lines.append(
            f"| {display} | {len(grouped_docs[secondary])} | [进入分类]({secondary}/README.md) |"
        )
    lines.extend(
        [
            "",
            "## 阅读建议",
            "- 先按二级分类筛选，再进入单算法文件查看速查表与详细说明。",
            "- 若关注跨类别对比，请回到根目录的 taxonomy.md 查看全量总表。",
            "",
        ]
    )
    return "\n".join(lines)


def render_secondary_readme(primary: str, secondary: str, docs: list[AlgorithmDoc]) -> str:
    lines = [
        f"# {secondary}",
        "",
        f"> 一级分类：{primary}",
        f"> 算法数量：{len(docs)}",
        "",
        "## 分类说明",
        f"该目录汇总 `{secondary}` 相关算法，统一按单算法文件组织，并通过下表提供快速检索入口。",
        "",
        "## 算法汇总表",
        "| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for doc in docs:
        lines.append(
            f"| [{doc.link_title}]({doc.path.name}) | {doc.task_type} | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 |"
        )
    lines.append("")
    return "\n".join(lines)


def render_taxonomy(docs: list[AlgorithmDoc]) -> str:
    grouped = build_category_map(docs)
    lines = [
        "# ML/DL 算法总索引",
        "",
        "## 一级分类总览",
        "| 一级分类 | 算法数量 | 链接 |",
        "|---|---:|---|",
    ]
    for primary in PRIMARY_ORDER:
        category_docs = grouped.get(primary, {})
        total = sum(len(entries) for entries in category_docs.values())
        lines.append(f"| {primary} | {total} | [进入](算法/{primary}/README.md) |")

    lines.extend(
        [
            "",
            "## 二级分类总览",
            "| 一级分类 | 二级分类 | 算法数量 | 链接 |",
            "|---|---|---:|---|",
        ]
    )
    for primary in PRIMARY_ORDER:
        for secondary in sorted(grouped.get(primary, {})):
            display = SECONDARY_DISPLAY_OVERRIDES.get((primary, secondary), secondary)
            lines.append(
                f"| {primary} | {display} | {len(grouped[primary][secondary])} | [进入](算法/{primary}/{secondary}/README.md) |"
            )

    lines.extend(
        [
            "",
            "## 全量算法总表",
            "| 算法 | 任务类型 | 适用场景 | 优点 | 缺点 | 解决办法 | 常用参数 | 选型建议 | 解释方法 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for doc in docs:
        lines.append(
            f"| [{doc.link_title}]({doc.taxonomy_link}) | {doc.task_type} | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 | 见单文件详情 |"
        )
    lines.append("")
    return "\n".join(lines)


def render_root_readme_cn(docs: list[AlgorithmDoc]) -> str:
    return "\n".join(
        [
            "# Machine Learning And Deep Learning 文档导航",
            "",
            "本目录现已重构为 **一个算法一个 Markdown 文件** 的组织方式。",
            "",
            "## 入口",
            "- [总分类索引](taxonomy.md)",
            "- [机器学习](算法/机器学习/README.md)",
            "- [深度学习](算法/深度学习/README.md)",
            "- [术语表](glossary.md)",
            "- [归档](归档/README.md)",
            "",
            f"- 当前算法文件数量：{len(docs)}",
            "",
            "## 阅读建议",
            "1. 先在 taxonomy.md 按一级分类 / 二级分类 / 任务类型查找。",
            "2. 再进入对应二级分类 README 查看汇总表。",
            "3. 最后进入具体算法文件查看速查表、选型建议与解释方法。",
            "",
        ]
    )


def render_root_readme_en(docs: list[AlgorithmDoc]) -> str:
    return "\n".join(
        [
            "# Machine Learning And Deep Learning Documentation",
            "",
            "This directory has been restructured into a **one algorithm per Markdown file** architecture.",
            "",
            "## Entry Points",
            "- [Global Taxonomy](taxonomy.md)",
            "- [Machine Learning](算法/机器学习/README.md)",
            "- [Deep Learning](算法/深度学习/README.md)",
            "- [Glossary](glossary.md)",
            "- [Archive](归档/README.md)",
            "",
            f"- Algorithm files: {len(docs)}",
            "",
        ]
    )


def write_if_changed(path: Path, content: str) -> bool:
    current = read_text(path) if path.exists() else None
    if current == content:
        return False
    write_text(path, content)
    return True


def rename_docs(docs: list[AlgorithmDoc]) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []
    for doc in docs:
        if not requires_canonical_rename(doc):
            continue
        target = doc.path.with_name(doc.canonical_filename)
        if target == doc.path:
            continue
        if target.exists():
            raise FileExistsError(f"Rename target already exists: {target}")
        renames.append((doc.path, target))
    for source, target in renames:
        source.rename(target)
    return renames


def regenerate_indexes(docs: list[AlgorithmDoc]) -> list[Path]:
    changed: list[Path] = []
    grouped = build_category_map(docs)

    for primary in PRIMARY_ORDER:
        primary_dir = ALGORITHM_ROOT / primary
        if write_if_changed(primary_dir / "README.md", render_primary_readme(primary, grouped.get(primary, {}))):
            changed.append(primary_dir / "README.md")
        for secondary, secondary_docs in grouped.get(primary, {}).items():
            target = primary_dir / secondary / "README.md"
            if write_if_changed(target, render_secondary_readme(primary, secondary, secondary_docs)):
                changed.append(target)

    if write_if_changed(TAXONOMY, render_taxonomy(docs)):
        changed.append(TAXONOMY)
    if write_if_changed(ROOT_README_CN, render_root_readme_cn(docs)):
        changed.append(ROOT_README_CN)
    if write_if_changed(ROOT_README, render_root_readme_en(docs)):
        changed.append(ROOT_README)

    return changed


def validate_links(path: Path) -> list[str]:
    errors: list[str] = []
    content = read_text(path)
    for link in LINK_PATTERN.findall(content):
        if link.startswith("http://") or link.startswith("https://"):
            continue
        target = (path.parent / link).resolve()
        if not target.exists():
            errors.append(f"Broken link in {path.relative_to(PROJECT_ROOT)} -> {link}")
    return errors


def extract_reference_items(text: str) -> list[str]:
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return []
    items: list[str] = []
    seen: set[str] = set()
    for line in match.group(0).splitlines()[1:]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        if stripped not in seen:
            seen.add(stripped)
            items.append(stripped)
    return items


def rewrite_reference_section(text: str, guidance: list[str]) -> str:
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return text

    existing_items = extract_reference_items(text)
    prefix = text[: match.start()].rstrip()
    lines = [
        "## 参考资料",
        "### 教材与综述",
    ]
    lines.extend(f"- {item}" for item in guidance)
    if existing_items:
        lines.extend(["", "### 扩展阅读"])
        lines.extend(existing_items)

    return prefix + "\n\n" + "\n".join(lines) + "\n"


def audit(docs: list[AlgorithmDoc]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[int] = set()
    for doc in docs:
        if doc.algo_id in seen_ids:
            errors.append(f"Duplicate algorithm id: {doc.algo_id:03d}")
        seen_ids.add(doc.algo_id)

        expected_secondary = doc.path.parent.name
        expected_primary = doc.path.parent.parent.name
        if normalize_secondary_name(expected_secondary) != normalize_secondary_name(doc.secondary):
            errors.append(
                f"Secondary category mismatch for {doc.relative_path}: metadata={doc.secondary}, folder={expected_secondary}"
            )
        if expected_primary != doc.primary:
            errors.append(
                f"Primary category mismatch for {doc.relative_path}: metadata={doc.primary}, folder={expected_primary}"
            )
        if requires_canonical_rename(doc) and doc.path.name != doc.canonical_filename:
            errors.append(
                f"Non-canonical filename for {doc.relative_path}: expected {doc.canonical_filename}"
            )

    files_to_check = [ROOT_README, ROOT_README_CN, TAXONOMY]
    files_to_check.extend(path for path in ALGORITHM_ROOT.rglob("README.md"))
    for readme in files_to_check:
        errors.extend(validate_links(readme))

    return errors


def reference_audit(docs: list[AlgorithmDoc]) -> tuple[int, int]:
    legacy_only = 0
    structured = 0
    for doc in docs:
        if "## 参考资料" not in doc.text:
            continue
        has_legacy = "[Legacy Source:" in doc.text
        has_structure = any(
            marker in doc.text
            for marker in ("### 原始论文", "### 教材参考", "### 教材与综述", "### 扩展阅读")
        )
        if has_legacy and not has_structure:
            legacy_only += 1
        if has_structure:
            structured += 1
    return legacy_only, structured


def enrich_references(docs: list[AlgorithmDoc]) -> list[Path]:
    changed: list[Path] = []
    for doc in docs:
        guidance = CATEGORY_REFERENCE_GUIDANCE.get((doc.primary, doc.secondary))
        if not guidance:
            continue
        updated = rewrite_reference_section(doc.text, guidance)
        if updated != doc.text:
            write_text(doc.path, updated)
            changed.append(doc.path)
    return changed


def run_sync() -> int:
    docs = load_docs()
    renames = rename_docs(docs)
    updated_docs = load_docs()
    changed_indexes = regenerate_indexes(updated_docs)

    print(f"Renamed files: {len(renames)}")
    for source, target in renames:
        print(f"  {source.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}")

    print(f"Regenerated index/readme files: {len(changed_indexes)}")
    for path in changed_indexes:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


def run_audit() -> int:
    docs = load_docs()
    errors = audit(docs)
    legacy_only, structured = reference_audit(docs)

    print(f"Algorithm documents: {len(docs)}")
    print(f"Canonical filename mismatches / index issues: {len(errors)}")
    print(f"Legacy-only reference sections: {legacy_only}")
    print(f"Structured reference sections: {structured}")

    if errors:
        print("\nDetailed audit findings:")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


def run_references() -> int:
    docs = load_docs()
    changed = enrich_references(docs)
    print(f"Reference sections updated: {len(changed)}")
    for path in changed:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain ML/DL algorithm documentation indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync", help="Rename algorithm files to canonical names and rebuild indexes.")
    subparsers.add_parser("audit", help="Validate filenames and generated navigation files.")
    subparsers.add_parser(
        "references",
        help="Upgrade reference sections with category-level textbook and survey sources from the implementation plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sync":
        return run_sync()
    if args.command == "audit":
        return run_audit()
    if args.command == "references":
        return run_references()
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
