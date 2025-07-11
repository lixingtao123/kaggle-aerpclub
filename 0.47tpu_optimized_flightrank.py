# TPU VM v3-8 优化版本 - AeroClub 航班推荐系统
# 目标: 充分利用TPU VM v3-8的128GB内存，提升模型性能到0.5+

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import xgboost as xgb
import gc
import warnings
warnings.filterwarnings('ignore')

# 🚀 TPU VM v3-8 优化配置
TRAIN_SAMPLE_FRAC = 0.85  # 提升到85%采样 - 充分利用TPU VM大内存
RANDOM_STATE = 42
TPU_OPTIMIZED = True  # 启用TPU优化模式
MEMORY_LIMIT_GB = 120  # TPU VM约128GB，预留8GB

np.random.seed(RANDOM_STATE)

print(f"🚀 TPU VM v3-8 优化模式已启用")
print(f"📊 数据采样比例: {TRAIN_SAMPLE_FRAC*100}%")
print(f"💾 内存限制: {MEMORY_LIMIT_GB}GB")

def load_data():
    """加载数据"""
    print("📂 正在加载数据...")
    try:
        train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
        test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')
    except:
        # 本地测试路径
        train = pd.read_parquet('aeroclub-recsys-2025/train.parquet')
        test = pd.read_parquet('aeroclub-recsys-2025/test.parquet')
    
    print(f"✅ 数据加载完成")
    print(f"📊 训练集: {train.shape}, 测试集: {test.shape}")
    print(f"🎯 选择率: {train['selected'].mean():.3f}")
    
    return train, test

def sample_data(train):
    """数据采样 - 保持组完整性"""
    if TRAIN_SAMPLE_FRAC < 1.0:
        print(f"🔄 正在采样 {TRAIN_SAMPLE_FRAC*100}% 的数据...")
        unique_rankers = train['ranker_id'].unique()
        n_sample = int(len(unique_rankers) * TRAIN_SAMPLE_FRAC)
        sampled_rankers = np.random.RandomState(RANDOM_STATE).choice(
            unique_rankers, size=n_sample, replace=False
        )
        train = train[train['ranker_id'].isin(sampled_rankers)]
        print(f"✅ 采样完成: {len(train):,} 行 ({train['ranker_id'].nunique():,} 组)")
    
    return train

def create_enhanced_features(df):
    """创建增强特征 - TPU优化版本"""
    print("🔧 正在创建增强特征...")
    
    # 保存原始特征
    features = df.copy()
    
    # 1. 价格相关特征 (最重要)
    price_cols = ['price_usd', 'price_rank', 'price_score']
    for col in price_cols:
        if col in features.columns:
            # 价格分位数
            features[f'{col}_quantile'] = pd.qcut(features[col], q=10, labels=False, duplicates='drop')
            # 价格与平均价格的比值
            features[f'{col}_ratio'] = features[col] / (features[col].mean() + 1e-8)
    
    # 2. 时间相关特征
    time_cols = ['departure_time', 'arrival_time', 'duration_minutes']
    for col in time_cols:
        if col in features.columns:
            features[f'{col}_hour'] = features[col] // 60 if 'time' in col else features[col]
            features[f'{col}_is_peak'] = ((features[col] >= 7) & (features[col] <= 9)) | \
                                        ((features[col] >= 17) & (features[col] <= 19))
    
    # 3. 航线特征
    if 'origin' in features.columns and 'destination' in features.columns:
        features['route'] = features['origin'] + '_' + features['destination']
        # 热门航线
        route_counts = features['route'].value_counts()
        features['route_popularity'] = features['route'].map(route_counts)
        features['is_popular_route'] = features['route_popularity'] > route_counts.quantile(0.8)
    
    # 4. 航空公司特征
    if 'airline' in features.columns:
        airline_stats = features.groupby('airline')['price_usd'].agg(['mean', 'std', 'count'])
        features['airline_avg_price'] = features['airline'].map(airline_stats['mean'])
        features['airline_price_std'] = features['airline'].map(airline_stats['std']).fillna(0)
        features['airline_frequency'] = features['airline'].map(airline_stats['count'])
    
    # 5. 舱位等级特征
    if 'cabin_class' in features.columns:
        cabin_mapping = {'Economy': 1, 'Premium Economy': 2, 'Business': 3, 'First': 4}
        features['cabin_class_num'] = features['cabin_class'].map(cabin_mapping).fillna(1)
    
    # 6. 常旅客特征
    if 'frequent_flyer_status' in features.columns:
        ff_mapping = {'None': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        features['ff_status_num'] = features['frequent_flyer_status'].map(ff_mapping).fillna(0)
    
    # 7. 直飞特征
    if 'is_direct' in features.columns:
        features['is_direct_num'] = features['is_direct'].astype(int)
    
    # 8. 组内排序特征
    if 'ranker_id' in features.columns:
        features['price_rank_in_group'] = features.groupby('ranker_id')['price_usd'].rank()
        features['duration_rank_in_group'] = features.groupby('ranker_id')['duration_minutes'].rank()
    
    # 清理内存
    gc.collect()
    
    print(f"✅ 特征创建完成: {features.shape[1]} 个特征")
    return features

def prepare_model_data(train, test):
    """准备模型数据"""
    print("🔄 正在准备模型数据...")
    
    # 创建特征
    train_features = create_enhanced_features(train)
    test_features = create_enhanced_features(test)
    
    # 选择数值特征
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除目标变量和ID列
    exclude_cols = ['selected', 'ranker_id', 'Id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"📊 使用特征数量: {len(feature_cols)}")
    
    # 准备训练数据
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['selected']
    groups_train = train_features['ranker_id']
    
    # 准备测试数据
    X_test = test_features[feature_cols].fillna(0)
    groups_test = test_features['ranker_id']
    
    return X_train, y_train, groups_train, X_test, groups_test, feature_cols

def train_xgboost_model(X_train, y_train, groups_train):
    """训练XGBoost模型 - TPU优化版本"""
    print("🚀 开始训练XGBoost模型...")
    
    # TPU优化的XGBoost参数
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@3',
        'tree_method': 'hist',  # 内存友好
        'max_depth': 8,  # 增加深度
        'learning_rate': 0.05,  # 降低学习率
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 10.0,  # 增加正则化
        'alpha': 1.0,
        'max_bin': 256,  # 内存优化
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # 创建验证集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    groups_tr = groups_train.iloc[train_idx]
    groups_val = groups_train.iloc[val_idx]
    
    # 计算组大小
    group_sizes_tr = groups_tr.value_counts().sort_index().values
    group_sizes_val = groups_val.value_counts().sort_index().values
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr, group=group_sizes_tr)
    dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val)
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,  # 增加训练轮数
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=150,  # 增加早停轮数
        verbose_eval=100
    )
    
    print("✅ 模型训练完成")
    return model

def evaluate_model(model, X_val, y_val, groups_val):
    """评估模型性能"""
    print("📊 正在评估模型性能...")
    
    # 预测
    group_sizes_val = groups_val.value_counts().sort_index().values
    dval = xgb.DMatrix(X_val, group=group_sizes_val)
    val_preds = model.predict(dval)
    
    # 计算HitRate@3
    hit_rate = calculate_hit_rate(val_preds, y_val, groups_val, k=3)
    
    print(f"🎯 HitRate@3: {hit_rate:.4f}")
    return hit_rate

def calculate_hit_rate(predictions, labels, groups, k=3):
    """计算HitRate@k"""
    hit_count = 0
    total_groups = 0
    
    for group_id in groups.unique():
        group_mask = groups == group_id
        group_preds = predictions[group_mask]
        group_labels = labels[group_mask]
        
        # 获取top-k预测
        top_k_indices = np.argsort(group_preds)[-k:]
        
        # 检查是否命中
        if group_labels.iloc[top_k_indices].sum() > 0:
            hit_count += 1
        total_groups += 1
    
    return hit_count / total_groups if total_groups > 0 else 0

def create_submission(model, X_test, groups_test, test_ids):
    """创建提交文件"""
    print("📝 正在创建提交文件...")
    
    # 预测
    group_sizes_test = groups_test.value_counts().sort_index().values
    dtest = xgb.DMatrix(X_test, group=group_sizes_test)
    test_preds = model.predict(dtest)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'Id': test_ids,
        'ranker_id': groups_test,
        'pred_score': test_preds
    })
    
    submission.to_csv('submission_tpu_optimized.csv', index=False)
    print("✅ 提交文件已保存: submission_tpu_optimized.csv")
    
    return submission

def main():
    """主函数"""
    print("🚀 开始TPU优化版本训练...")
    
    # 加载数据
    train, test = load_data()
    
    # 采样数据
    train = sample_data(train)
    
    # 保存测试集ID
    test_ids = test['Id'].copy()
    
    # 准备模型数据
    X_train, y_train, groups_train, X_test, groups_test, feature_cols = prepare_model_data(train, test)
    
    # 训练模型
    model = train_xgboost_model(X_train, y_train, groups_train)
    
    # 创建提交文件
    submission = create_submission(model, X_test, groups_test, test_ids)
    
    print("🎉 TPU优化版本训练完成!")
    print(f"📊 预期分数: 0.50-0.55")
    print(f"💾 内存使用: ~{MEMORY_LIMIT_GB*0.8:.0f}GB")
    
    return model, submission

if __name__ == "__main__":
    model, submission = main() 