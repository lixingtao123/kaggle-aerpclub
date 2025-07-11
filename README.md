# AeroClub RecSys 2025 - 航班推荐系统

这个项目是Kaggle AeroClub推荐系统挑战赛的解决方案，使用XGBoost排序模型来预测用户选择特定航班选项的概率。

## 项目描述

- **竞赛**: AeroClub RecSys 2025
- **目标**: 预测用户在航班搜索结果中选择特定选项的概率
- **方法**: XGBoost排序模型 (rank:pairwise)
- **评估指标**: NDCG@3, HitRate@3, LogLoss

## 文件结构

- `xgboost-ranker-baseline-flightrank-2025.ipynb` - 主要的Jupyter notebook，包含完整的模型训练和评估流程
- `jsons_structure.md` - 数据结构的详细说明
- `train.parquet` - 训练数据 (需要从Kaggle下载)
- `test.parquet` - 测试数据 (需要从Kaggle下载)
- `sample_submission.parquet` - 提交样例

## 主要特性

### 特征工程
- 价格相关特征 (价格排名、税率、价格比率等)
- 时间特征 (出发/到达时间、工作日、商务时间等)
- 航班特征 (直飞、转机次数、行李、费用等)
- 用户特征 (VIP状态、常旅客计划、企业关税等)
- 排名特征 (价格排名、时长排名等)

### 模型配置
- **算法**: XGBoost with rank:pairwise objective
- **评估指标**: NDCG@3
- **早停**: 100轮
- **最大深度**: 8
- **学习率**: 0.05

## 使用方法

1. 从Kaggle下载数据集
2. 运行Jupyter notebook: `xgboost-ranker-baseline-flightrank-2025.ipynb`
3. 生成提交文件: `submission.csv`

## 环境要求

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## 性能指标

- **HitRate@3**: 0.5042
- **LogLoss**: 0.6871
- **Top-1 Accuracy**: 0.3520

## 作者

[lixingtao123](https://github.com/lixingtao123)

## 许可证

MIT License 