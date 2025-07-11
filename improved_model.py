# %% [markdown]
# # AeroClub RecSys 2025 - 改进的XGBoost排序模型
# 
# 这个项目是对原始基线模型的全面改进，包含以下功能：
# - 完整的特征工程
# - 模型训练和评估
# - 错误处理和内存优化
# - 详细的中文注释

# %% [markdown]
# ## 1. 导入必要的库

# %%
"""
AeroClub RecSys 2025 - 改进的XGBoost排序模型

项目目标：
1. 添加详细的中文注释，提高代码可读性
2. 修复原始代码中的潜在问题
3. 为后续的特征工程和模型优化打下基础
4. 实现完整的机器学习流程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2. 全局配置

# %%
# 全局参数配置
TRAIN_SAMPLE_FRAC = 0.5  # 训练数据采样比例，用于快速迭代
RANDOM_STATE = 42        # 随机种子，确保结果可重现

# 设置随机种子
np.random.seed(RANDOM_STATE)

# 显示设置，便于数据查看
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

print("全局配置完成")
print(f"训练数据采样比例: {TRAIN_SAMPLE_FRAC}")
print(f"随机种子: {RANDOM_STATE}")

# %% [markdown]
# ## 3. 航班排序模型类定义

# %%
class FlightRankingModel:
    """
    航班推荐排序模型
    使用XGBoost实现的排序学习模型，专门用于航班选择预测
    """
    
    def __init__(self, train_sample_frac: float = 0.5, random_state: int = 42, 
                 chunk_size: int = 100000, use_chunked_loading: bool = False):
        """
        初始化模型
        
        Args:
            train_sample_frac: 训练数据采样比例
            random_state: 随机种子
            chunk_size: 分块大小，用于大数据集处理
            use_chunked_loading: 是否使用分块加载
        """
        self.train_sample_frac = train_sample_frac
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.use_chunked_loading = use_chunked_loading
        self.model = None
        self.feature_cols = []
        self.cat_features = []
        
        # 设置随机种子
        np.random.seed(random_state)
        
        # 全局配置
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.max_rows', 20)
        
        print(f"FlightRankingModel 初始化完成")
        print(f"   - 训练采样比例: {train_sample_frac}")
        print(f"   - 随机种子: {random_state}")
        print(f"   - 分块大小: {chunk_size:,}")
        print(f"   - 分块加载: {'启用' if use_chunked_loading else '禁用'}")

# %% [markdown]
# ## 4. 数据加载功能

# %%
    def load_data_chunked(self, file_path: str) -> pd.DataFrame:
        """
        分块加载大型parquet文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            合并后的DataFrame
        """
        print(f"开始分块加载数据: {file_path}")
        chunks = []
        
        try:
            # 使用pandas分块读取
            chunk_iter = pd.read_parquet(file_path, chunksize=self.chunk_size)
            
            for i, chunk in enumerate(chunk_iter):
                print(f"  处理分块 {i+1}: {len(chunk):,} 行")
                chunks.append(chunk)
                
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1} 个分块，正在合并...")
                    # 可以在这里添加内存清理逻辑
                    import gc
                    gc.collect()
            
            print(f"合并 {len(chunks)} 个分块...")
            df = pd.concat(chunks, ignore_index=True)
            print(f"分块加载完成: {len(df):,} 行")
            
            return df
            
        except Exception as e:
            print(f"分块加载失败: {e}")
            # 回退到常规加载
            print("回退到常规加载方式...")
            return pd.read_parquet(file_path)

    def load_data(self, train_path: str = 'train.parquet', test_path: str = 'test.parquet') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载训练和测试数据
        
        Args:
            train_path: 训练数据文件路径
            test_path: 测试数据文件路径
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 训练数据和测试数据
        """
        print("开始加载数据...")
        
        try:
            if self.use_chunked_loading:
                # 分块加载
                train_data = self.load_data_chunked(train_path)
                test_data = self.load_data_chunked(test_path)
            else:
                # 常规加载
                print(f"加载训练数据: {train_path}")
                train_data = pd.read_parquet(train_path)
                print(f"加载测试数据: {test_path}")
                test_data = pd.read_parquet(test_path)
            
            print(f"数据加载完成")
            print(f"  训练数据形状: {train_data.shape}")
            print(f"  测试数据形状: {test_data.shape}")
            print(f"  训练集中唯一的ranker_id数量: {train_data['ranker_id'].nunique():,}")
            print(f"  目标变量(selected)的平均值: {train_data['selected'].mean():.3f}")
            
            return train_data, test_data
            
        except FileNotFoundError as e:
            print(f"文件未找到错误: {e}")
            print("请检查文件路径是否正确！")
            return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            print(f"数据加载出错: {e}")
            return pd.DataFrame(), pd.DataFrame()

# %% [markdown]
# ## 5. 数据采样功能

# %%
    def sample_data(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        对训练数据进行采样（按ranker_id分组采样）
        
        Args:
            train_df: 原始训练数据
            
        Returns:
            pd.DataFrame: 采样后的训练数据
        """
        if self.train_sample_frac >= 1.0:
            print("🔄 使用全部训练数据，无需采样")
            return train_df
        
        print(f"开始数据采样（采样比例: {self.train_sample_frac}）")
        
        # 获取所有唯一的ranker_id
        unique_rankers = train_df['ranker_id'].unique()
        n_sample = int(len(unique_rankers) * self.train_sample_frac)
        
        # 随机选择ranker_id进行采样
        # 这样做可以确保同一个ranker_id下的所有选项要么全部包含，要么全部排除，避免数据泄露
        sampled_rankers = np.random.RandomState(self.random_state).choice(
            unique_rankers, size=n_sample, replace=False
        )
        
        # 根据选中的ranker_id过滤数据
        sampled_df = train_df[train_df['ranker_id'].isin(sampled_rankers)]
        
        print(f"数据采样完成")
        print(f"   - 采样前: {len(train_df):,} 行，{train_df['ranker_id'].nunique():,} 组")
        print(f"   - 采样后: {len(sampled_df):,} 行，{sampled_df['ranker_id'].nunique():,} 组")
        
        return sampled_df

# %% [markdown]
# ## 6. 时间转换辅助函数

# %%
    def hms_to_minutes(self, s: pd.Series) -> np.ndarray:
        """
        将 'HH:MM:SS' 格式的时间转换为分钟数（忽略秒）
        
        Args:
            s: 包含时间字符串的Series
            
        Returns:
            np.ndarray: 转换后的分钟数数组
        """
        mask = s.notna()
        out = np.zeros(len(s), dtype=float)
        if mask.any():
            parts = s[mask].astype(str).str.split(':', expand=True)
            out[mask] = (
                pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
                + pd.to_numeric(parts[1], errors="coerce").fillna(0)
            )
        return out

# %% [markdown]
# ## 7. 特征工程主函数

# %%
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征工程
        
        Args:
            df: 输入数据
            
        Returns:
            包含新特征的数据
        """
        print("开始特征工程...")
        
        # 如果数据量很大，使用分块处理
        if len(df) > self.chunk_size * 2:
            print(f"数据量较大 ({len(df):,} 行)，使用分块处理...")
            return self.process_data_in_chunks(df, self._create_features_single_chunk)
        else:
            return self._create_features_single_chunk(df)
    
    def _create_features_single_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        单个分块的特征工程处理
        
        Args:
            df: 输入数据块
            
        Returns:
            包含新特征的数据块
        """
        df = df.copy()
        
        # 处理时间格式数据
        dur_cols = (
            ["legs0_duration", "legs1_duration"]
            + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
        )
        for col in dur_cols:
            if col in df.columns:
                df[col] = self.hms_to_minutes(df[col])

        # 特征容器
        feat = {}

        # 价格特征
        print("   创建价格特征...")
        feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
        feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
        feat["log_price"] = np.log1p(df["totalPrice"])

        # 时长特征
        print("   创建时长特征...")
        df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
        feat["duration_ratio"] = np.where(
            df["legs1_duration"].fillna(0) > 0,
            df["legs0_duration"] / (df["legs1_duration"] + 1),
            1.0,
        )

        # 航段数量特征
        print("   创建航段特征...")
        for leg in (0, 1):
            seg_count = 0
            for seg in range(4):
                col = f"legs{leg}_segments{seg}_duration"
                if col in df.columns:
                    seg_count += df[col].notna().astype(int)
                else:
                    break
            feat[f"n_segments_leg{leg}"] = seg_count
        
        feat["total_segments"] = feat["n_segments_leg0"] + feat["n_segments_leg1"]

        # 行程类型检测
        feat["is_one_way"] = (
            df["legs1_duration"].isna() | 
            (df["legs1_duration"] == 0) |
            df["legs1_segments0_departureFrom_airport_iata"].isna()
        ).astype(int)
        
        feat["has_return"] = (1 - feat["is_one_way"]).astype(int)

        # 排名特征
        print("   创建排名特征...")
        if len(df) > 0 and 'ranker_id' in df.columns:
            try:
                grp = df.groupby("ranker_id")
                feat["price_rank"] = grp["totalPrice"].rank()
                feat["price_pct_rank"] = grp["totalPrice"].rank(pct=True)
                feat["duration_rank"] = grp["total_duration"].rank()
                feat["is_cheapest"] = (grp["totalPrice"].transform("min") == df["totalPrice"]).astype(int)
                feat["is_most_expensive"] = (grp["totalPrice"].transform("max") == df["totalPrice"]).astype(int)
                feat["price_from_median"] = grp["totalPrice"].transform(
                    lambda x: (x - x.median()) / (x.std() + 1)
                )
            except Exception as e:
                print(f"   警告: 排名特征创建失败: {e}")
                # 创建默认值
                feat["price_rank"] = 1
                feat["price_pct_rank"] = 0.5
                feat["duration_rank"] = 1
                feat["is_cheapest"] = 0
                feat["is_most_expensive"] = 0
                feat["price_from_median"] = 0

        # 常旅客特征
        print("   创建常旅客特征...")
        ff = df["frequentFlyer"].fillna("").astype(str)
        feat["n_ff_programs"] = ff.str.count("/") + (ff != "")
        
        carrier_cols = ["legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"]
        present_airlines = set()
        for col in carrier_cols:
            if col in df.columns:
                present_airlines.update(df[col].dropna().unique())
        
        for al in ["SU", "S7", "U6", "TK"]:
            if al in present_airlines:
                feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)
        
        feat["ff_matches_carrier"] = 0
        for al in ["SU", "S7", "U6", "TK"]:
            if f"ff_{al}" in feat and "legs0_segments0_marketingCarrier_code" in df.columns:
                feat["ff_matches_carrier"] |= (
                    (feat.get(f"ff_{al}", 0) == 1) & 
                    (df["legs0_segments0_marketingCarrier_code"] == al)
                ).astype(int)

        # 二元标志
        feat["is_vip_freq"] = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int)
        feat["has_corporate_tariff"] = (~df["corporateTariffCode"].isna()).astype(int)

        # 行李和费用
        print("   创建行李和费用特征...")
        feat["baggage_total"] = (
            df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
            + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
        )
        feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
        feat["total_fees"] = (
            df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
        )
        feat["has_fees"] = (feat["total_fees"] > 0).astype(int)
        feat["fee_rate"] = feat["total_fees"] / (df["totalPrice"] + 1)

        # 时间特征
        print("   创建时间特征...")
        for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                feat[f"{col}_hour"] = dt.dt.hour.fillna(12)
                feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
                h = dt.dt.hour.fillna(12)
                feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)

        # 直飞检测
        print("   创建直飞航班特征...")
        feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
        feat["is_direct_leg1"] = np.where(
            feat["is_one_way"] == 1,
            0,
            (feat["n_segments_leg1"] == 1).astype(int)
        )
        feat["both_direct"] = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype(int)

        # 最便宜直飞
        if len(df) > 0 and 'ranker_id' in df.columns:
            try:
                df["_is_direct"] = feat["is_direct_leg0"] == 1
                direct_groups = df[df["_is_direct"]].groupby("ranker_id")["totalPrice"]
                if len(direct_groups) > 0:
                    direct_min_price = direct_groups.min()
                    feat["is_direct_cheapest"] = (
                        df["_is_direct"] & 
                        (df["totalPrice"] == df["ranker_id"].map(direct_min_price))
                    ).astype(int)
                else:
                    feat["is_direct_cheapest"] = 0
                df.drop(columns="_is_direct", inplace=True)
            except Exception as e:
                print(f"   警告: 直飞最便宜特征创建失败: {e}")
                feat["is_direct_cheapest"] = 0

        # 其他特征
        print("   创建其他特征...")
        feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)
        
        if len(df) > 0 and 'ranker_id' in df.columns:
            try:
                feat["group_size"] = df.groupby("ranker_id")["Id"].transform("count")
                feat["group_size_log"] = np.log1p(feat["group_size"])
            except Exception as e:
                print(f"   警告: 组大小特征创建失败: {e}")
                feat["group_size"] = 1
                feat["group_size_log"] = 0
        else:
            feat["group_size"] = 1
            feat["group_size_log"] = 0
        
        if "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
        else:
            feat["is_major_carrier"] = 0
        
        popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"}
        feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)
        
        feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
        feat["cabin_class_diff"] = (
            df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
        )

        # 合并新特征
        df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)

        # 处理缺失值
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(0)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("missing")

        print(f"特征工程完成，共创建 {len(feat)} 个新特征")
        return df

# %% [markdown]
# ## 8. 特征选择功能

# %%
    def select_features(self, train_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        选择用于模型训练的特征
        
        Args:
            train_df: 训练数据
            
        Returns:
            Tuple[List[str], List[str]]: 所有特征列表和分类特征列表
        """
        print("开始特征选择...")
        
        # 分类特征
        cat_features = [
            'nationality', 'searchRoute', 'corporateTariffCode',
            # Leg 0 segments 0-1
            'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
            'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
            'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
            'legs0_segments0_flightNumber',
            'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
            'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
            'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
            'legs0_segments1_flightNumber',
            # Leg 1 segments 0-1
            'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
            'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
            'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
            'legs1_segments0_flightNumber',
            'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
            'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
            'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
            'legs1_segments1_flightNumber'
        ]

        # 需要排除的列（无信息或有问题的列）
        exclude_cols = [
            'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
            'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
            'miniRules0_percentage', 'miniRules1_percentage',  # >90% 缺失
            'frequentFlyer',  # 已处理
            # 排除常量或接近常量的列
            'bySelf', 'pricingInfo_passengerCount',
            # 排除行李重量计量类型列（可能是常量）
            'legs0_segments0_baggageAllowance_weightMeasurementType',
            'legs0_segments1_baggageAllowance_weightMeasurementType',
            'legs1_segments0_baggageAllowance_weightMeasurementType',
            'legs1_segments1_baggageAllowance_weightMeasurementType',
            # 排除数据中不存在的航空公司的常旅客特征
            'ff_DP', 'ff_UT', 'ff_EK', 'ff_N4', 'ff_5N', 'ff_LH'
        ]

        # 排除segment 2-3列（>98%缺失）
        for leg in [0, 1]:
            for seg in [2, 3]:
                for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                              'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                              'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                              'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
                    exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        cat_features_final = [col for col in cat_features if col in feature_cols]

        print(f"特征选择完成")
        print(f"   - 总特征数: {len(feature_cols)}")
        print(f"   - 分类特征数: {len(cat_features_final)}")
        
        self.feature_cols = feature_cols
        self.cat_features = cat_features_final
        
        return feature_cols, cat_features_final

# %% [markdown]
# ## 9. XGBoost数据预处理

# %%
    def prepare_xgb_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        为XGBoost准备数据（标签编码分类特征）
        
        Args:
            train_df: 训练数据
            test_df: 测试数据
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 处理后的训练、验证、测试数据
        """
        print("🔄 为XGBoost准备数据...")
        
        # 分离特征和目标变量
        X_train = train_df[self.feature_cols].copy()
        y_train = train_df['selected']
        groups_train = train_df['ranker_id']
        X_test = test_df[self.feature_cols].copy()
        groups_test = test_df['ranker_id']

        # 训练/验证集分割（按组分割）
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]

        # 标签编码分类特征
        print("   🏷️ 对分类特征进行标签编码...")
        for col in self.cat_features:
            if col in X_tr.columns:
                # 从所有数据创建映射
                unique_vals = pd.concat([X_tr[col], X_val[col], X_test[col]]).unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                
                X_tr[col] = X_tr[col].map(mapping).fillna(-1).astype(int)
                X_val[col] = X_val[col].map(mapping).fillna(-1).astype(int)
                X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

        # 创建数据分割信息
        train_split = pd.concat([
            X_tr,
            pd.DataFrame({
                'selected': y_tr,
                'ranker_id': groups_tr
            }, index=X_tr.index)
        ], axis=1)
        
        val_split = pd.concat([
            X_val,
            pd.DataFrame({
                'selected': y_val,
                'ranker_id': groups_val
            }, index=X_val.index)
        ], axis=1)
        
        test_split = pd.concat([
            X_test,
            pd.DataFrame({
                'ranker_id': groups_test
            }, index=X_test.index)
        ], axis=1)

        print(f"数据准备完成")
        print(f"   - 训练集: {len(train_split):,} 行")
        print(f"   - 验证集: {len(val_split):,} 行")
        print(f"   - 测试集: {len(test_split):,} 行")

        return train_split, val_split, test_split

# %% [markdown]
# ## 10. 模型训练功能

# %%
    def get_memory_usage(self) -> str:
        """获取当前内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        return f"{memory_mb:.1f} MB"
    
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame的内存使用
        
        Args:
            df: 输入DataFrame
            
        Returns:
            内存优化后的DataFrame
        """
        print(f"内存优化前: {self.get_memory_usage()}")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 优化数值列
        for col in df.select_dtypes(include=['int64']).columns:
            col_max = df[col].max()
            col_min = df[col].min()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # 优化浮点数列
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # 优化字符串列
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"内存优化: {original_memory:.1f} MB → {optimized_memory:.1f} MB "
              f"(节省 {(original_memory - optimized_memory) / original_memory * 100:.1f}%)")
        print(f"内存优化后: {self.get_memory_usage()}")
        
        return df

    def train_model(self, train_split: pd.DataFrame, val_split: pd.DataFrame) -> xgb.Booster:
        """
        训练XGBoost模型（支持分块训练）
        
        Args:
            train_split: 训练数据
            val_split: 验证数据
            
        Returns:
            训练好的XGBoost模型
        """
        print("开始训练XGBoost模型...")
        print(f"训练数据大小: {len(train_split):,} 行")
        print(f"验证数据大小: {len(val_split):,} 行")
        
        # 内存优化
        if len(train_split) > self.chunk_size:
            print("执行内存优化...")
            train_split = self.optimize_memory(train_split)
            val_split = self.optimize_memory(val_split)
        
        # 准备特征和标签
        X_train = train_split[self.feature_cols]
        y_train = train_split['selected']
        groups_train = train_split['ranker_id']
        
        X_val = val_split[self.feature_cols]
        y_val = val_split['selected']
        groups_val = val_split['ranker_id']
        
        # 创建组大小
        group_sizes_train = groups_train.value_counts().sort_index().values
        group_sizes_val = groups_val.value_counts().sort_index().values
        
        # 检查数据大小，决定是否分块训练
        if len(train_split) > self.chunk_size * 5:
            print(f"数据量很大 ({len(train_split):,} 行)，考虑使用增量训练...")
            return self._train_model_incremental(X_train, y_train, group_sizes_train, 
                                                X_val, y_val, group_sizes_val)
        else:
            return self._train_model_standard(X_train, y_train, group_sizes_train,
                                            X_val, y_val, group_sizes_val)
    
    def _train_model_standard(self, X_train, y_train, group_sizes_train,
                             X_val, y_val, group_sizes_val) -> xgb.Booster:
        """标准XGBoost训练"""
        print("使用标准训练模式...")
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, group=group_sizes_train)
        dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val)
        
        # XGBoost参数
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            'max_depth': 6,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 10.0,
            'learning_rate': 0.1,
            'seed': self.random_state,
            'n_jobs': -1
        }
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def _train_model_incremental(self, X_train, y_train, group_sizes_train,
                                X_val, y_val, group_sizes_val) -> xgb.Booster:
        """增量XGBoost训练（对于大数据集）"""
        print("使用增量训练模式...")
        print("注意: 当前XGBoost版本可能不完全支持增量学习")
        
        # 对于非常大的数据集，我们使用数据采样策略
        sample_size = min(len(X_train), self.chunk_size * 3)
        if sample_size < len(X_train):
            print(f"数据量过大，采样到 {sample_size:,} 行进行训练")
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            
            X_train_sampled = X_train.iloc[sample_idx]
            y_train_sampled = y_train.iloc[sample_idx]
            
            # 重新计算组大小
            groups_sampled = X_train_sampled.index.map(
                lambda x: X_train.iloc[x:x+1]['ranker_id'].iloc[0] if x < len(X_train) else 'unknown'
            )
            group_sizes_sampled = pd.Series(groups_sampled).value_counts().values
            
            return self._train_model_standard(X_train_sampled, y_train_sampled, group_sizes_sampled,
                                            X_val, y_val, group_sizes_val)
        else:
            return self._train_model_standard(X_train, y_train, group_sizes_train,
                                            X_val, y_val, group_sizes_val)

# %% [markdown]
# ## 11. 模型评估功能

# %%
    def sigmoid(self, x: np.ndarray, scale: float = 10.0) -> np.ndarray:
        """将分数转换为概率（使用sigmoid函数）"""
        return 1 / (1 + np.exp(-x / scale))

    def calculate_hitrate_at_k(self, df: pd.DataFrame, k: int = 3) -> float:
        """计算HitRate@k（针对超过10个选项的组）"""
        hits = []
        for ranker_id, group in df.groupby('ranker_id'):
            if len(group) > 10:
                top_k = group.nlargest(k, 'pred')
                hit = (top_k['selected'] == 1).any()
                hits.append(hit)
        return np.mean(hits) if hits else 0.0

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, model_name: str = "模型") -> Tuple[pd.DataFrame, float]:
        """
        评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测分数
            groups: 分组信息
            model_name: 模型名称
            
        Returns:
            Tuple[pd.DataFrame, float]: 评估结果DataFrame和HitRate@3
        """
        print(f"评估{model_name}性能...")
        
        df = pd.DataFrame({
            'ranker_id': groups,
            'pred': y_pred,
            'selected': y_true
        })
        
        # 获取每组的最高预测
        top_preds = df.loc[df.groupby('ranker_id')['pred'].idxmax()]
        top_preds['prob'] = self.sigmoid(top_preds['pred'])
        
        # 计算指标
        logloss = log_loss(top_preds['selected'], top_preds['prob'])
        hitrate_at_3 = self.calculate_hitrate_at_k(df, k=3)
        accuracy = (top_preds['selected'] == 1).mean()
        
        print(f"{model_name} 验证指标:")
        print(f"   HitRate@3 (组大小>10): {hitrate_at_3:.4f}")
        print(f"   LogLoss:                {logloss:.4f}")
        print(f"   Top-1 准确率:           {accuracy:.4f}")
        
        return df, hitrate_at_3

# %% [markdown]
# ## 12. 特征重要性分析

# %%
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            pd.DataFrame: 特征重要性排序
        """
        if not hasattr(self, 'model') or self.model is None:
            print("模型未训练，无法获取特征重要性")
            return pd.DataFrame()
        
        print("分析特征重要性...")
        
        # 获取XGBoost特征重要性
        importance = self.model.get_score(importance_type='gain')
        
        # 转换为DataFrame
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        print(f"特征重要性分析完成，共 {len(importance_df)} 个特征")
        print("Top 20 重要特征:")
        print(importance_df.head(20).to_string(index=False))
        
        return importance_df

# %% [markdown]
# ## 13. 生成预测结果

# %%
    def generate_predictions(self, test_split: pd.DataFrame) -> pd.DataFrame:
        """
        生成测试集预测结果
        
        Args:
            test_split: 测试数据
            
        Returns:
            pd.DataFrame: 提交格式的预测结果
        """
        if not hasattr(self, 'model') or self.model is None:
            print("模型未训练，无法生成预测")
            return pd.DataFrame()
        
        print("生成测试集预测...")
        
        # 准备测试数据
        X_test = test_split[self.feature_cols]
        groups_test = test_split['ranker_id']
        
        # 创建组大小
        group_sizes_test = test_split.groupby('ranker_id').size().values
        dtest = xgb.DMatrix(X_test, group=group_sizes_test)
        
        # 生成预测
        test_preds = self.model.predict(dtest)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': test_split.index,  # 使用原始的Id
            'ranker_id': groups_test,
            'pred_score': test_preds
        })
        
        # 根据ranker_id分组，按预测分数排名
        submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
            ascending=False, method='first'
        ).astype(int)
        
        # 只保留需要的列
        final_submission = submission[['Id', 'ranker_id', 'selected']].copy()
        
        print(f"预测生成完成，形状: {final_submission.shape}")
        
        return final_submission

# %% [markdown]
# ## 14. 完整的机器学习流程

# %%
    def run_full_pipeline(self, train_path: str = 'train.parquet', test_path: str = 'test.parquet') -> pd.DataFrame:
        """
        运行完整的机器学习流程
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
            
        Returns:
            pd.DataFrame: 最终的预测结果
        """
        print("开始完整的机器学习流程...")
        print("="*60)
        
        # 1. 加载数据
        train_df, test_df = self.load_data(train_path, test_path)
        if train_df.empty or test_df.empty:
            print("数据加载失败，终止流程")
            return pd.DataFrame()
        
        # 2. 数据采样
        train_df = self.sample_data(train_df)
        
        # 3. 特征工程
        print("\n" + "="*60)
        train_df = self.create_features(train_df)
        test_df = self.create_features(test_df)
        
        # 4. 特征选择
        print("\n" + "="*60)
        self.select_features(train_df)
        
        # 5. 数据预处理
        print("\n" + "="*60)
        train_split, val_split, test_split = self.prepare_xgb_data(train_df, test_df)
        
        # 6. 模型训练
        print("\n" + "="*60)
        self.train_model(train_split, val_split)
        
        # 7. 模型评估
        print("\n" + "="*60)
        val_preds = self.model.predict(xgb.DMatrix(
            val_split[self.feature_cols],
            group=val_split.groupby('ranker_id').size().values
        ))
        self.evaluate_model(val_split['selected'].values, val_preds, val_split['ranker_id'], "XGBoost")
        
        # 8. 特征重要性
        print("\n" + "="*60)
        self.get_feature_importance()
        
        # 9. 生成预测
        print("\n" + "="*60)
        submission = self.generate_predictions(test_split)
        
        # 10. 保存结果
        if not submission.empty:
            submission.to_csv('submission.csv', index=False)
            print(f"💾 预测结果已保存到 submission.csv")
        
        print("\n" + "="*60)
        print("完整的机器学习流程完成！")
        
        return submission

# %% [markdown]
# ## 15. 快速运行函数

# %%
def quick_run_large_dataset(train_path: str, test_path: str, output_path: str = 'submission.csv', 
                           sample_frac: float = 0.3, chunk_size: int = 50000, random_state: int = 42):
    """
    大数据集快速运行函数（内存优化版本）
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径  
        output_path: 输出文件路径
        sample_frac: 采样比例（建议0.1-0.5）
        chunk_size: 分块大小
        random_state: 随机种子
        
    Returns:
        pd.DataFrame: 提交结果
    """
    print("=== 大数据集优化模式 ===")
    print(f"内存优化模式")
    print(f"文件配置:")
    print(f"   训练数据: {train_path}")
    print(f"   测试数据: {test_path}")
    print(f"   输出文件: {output_path}")
    print(f"   采样比例: {sample_frac}")
    print(f"   分块大小: {chunk_size:,}")
    print()
    
    try:
        # 创建模型（启用分块加载和内存优化）
        model = FlightRankingModel(
            train_sample_frac=sample_frac,
            random_state=random_state,
            chunk_size=chunk_size,
            use_chunked_loading=True  # 启用分块加载
        )
        
        # 运行流程
        print("开始大数据集处理流程...")
        submission = model.run_full_pipeline(train_path, test_path)
        
        if not submission.empty:
            # 保存结果
            submission[['Id', 'selected']].to_csv(output_path, index=False)
            print(f"\n运行完成！")
            print(f"结果已保存到: {output_path}")
            print(f"生成了 {len(submission):,} 行预测结果")
            return submission
        else:
            print("运行失败")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def quick_run(train_path: str, test_path: str, output_path: str = 'submission.csv', 
              sample_frac: float = 0.5, random_state: int = 42):
    """
    快速运行函数
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径  
        output_path: 输出文件路径
        sample_frac: 采样比例
        random_state: 随机种子
        
    Returns:
        pd.DataFrame: 提交结果
    """
    print("快速运行模式")
    print(f"文件配置:")
    print(f"   训练数据: {train_path}")
    print(f"   测试数据: {test_path}")
    print(f"   输出文件: {output_path}")
    print(f"   采样比例: {sample_frac}")
    print()
    
    try:
        # 创建模型
        model = FlightRankingModel(
            train_sample_frac=sample_frac,
            random_state=random_state
        )
        
        # 运行流程
        submission = model.run_full_pipeline(train_path, test_path)
        
        if not submission.empty:
            # 保存结果
            submission[['Id', 'selected']].to_csv(output_path, index=False)
            print(f"\n运行完成！")
            print(f"结果已保存到: {output_path}")
            print(f"生成了 {len(submission):,} 行预测结果")
            return submission
        else:
            print("运行失败")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"运行出错: {e}")
        return pd.DataFrame()

# %% [markdown]
# ## 16. 主函数和执行

# %%
def main():
    """主函数"""
    
    # ===== 文件路径配置区域 - 在这里修改您的文件路径 =====
    TRAIN_FILE_PATH = 'train.parquet'        # 修改这里设置训练文件路径
    TEST_FILE_PATH = 'test.parquet'          # 修改这里设置测试文件路径
    OUTPUT_FILE_PATH = 'submission.csv'      # 修改这里设置输出文件路径
    
    # 性能配置
    USE_LARGE_DATASET_MODE = True           # 是否使用大数据集优化模式
    CHUNK_SIZE = 50000                      # 分块大小
    SAMPLE_FRAC = 0.3                       # 采样比例（大数据集模式下建议0.1-0.5）
    # ==========================================================
    
    try:
        print("=== AeroClub RecSys 2025 - 改进的XGBoost排序模型 ===")
        print(f"使用的文件路径:")
        print(f"   训练数据: {TRAIN_FILE_PATH}")
        print(f"   测试数据: {TEST_FILE_PATH}")
        print(f"   输出文件: {OUTPUT_FILE_PATH}")
        
        if USE_LARGE_DATASET_MODE:
            print(f"   运行模式: 大数据集优化模式")
            print(f"   分块大小: {CHUNK_SIZE:,}")
            print(f"   采样比例: {SAMPLE_FRAC}")
        else:
            print(f"   运行模式: 标准模式")
            print(f"   采样比例: {TRAIN_SAMPLE_FRAC}")
        print()
        
        if USE_LARGE_DATASET_MODE:
            # 使用大数据集优化模式
            submission = quick_run_large_dataset(
                TRAIN_FILE_PATH, 
                TEST_FILE_PATH, 
                OUTPUT_FILE_PATH,
                sample_frac=SAMPLE_FRAC,
                chunk_size=CHUNK_SIZE,
                random_state=RANDOM_STATE
            )
        else:
            # 创建模型实例
            model = FlightRankingModel(
                train_sample_frac=TRAIN_SAMPLE_FRAC,
                random_state=RANDOM_STATE
            )
            
            # 运行完整流程
            submission = model.run_full_pipeline(TRAIN_FILE_PATH, TEST_FILE_PATH)
        
        if not submission.empty:
            # 保存到指定路径
            submission[['Id', 'selected']].to_csv(OUTPUT_FILE_PATH, index=False)
            print(f"\n任务完成！生成了 {len(submission)} 行预测结果")
            print(f"输出文件: {OUTPUT_FILE_PATH}")
        else:
            print("\n任务失败，请检查数据文件和错误信息")
            
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()

# %%
if __name__ == "__main__":
    main()

# %% [markdown]
# ## 17. 结语
# 
# 这个改进的模型包含了以下功能：
# 
# ### 主要改进点：
# 1. **完整的面向对象设计** - 使用 `FlightRankingModel` 类封装所有功能
# 2. **全面的特征工程** - 价格、时间、排名、用户特征等
# 3. **完善的错误处理** - 文件不存在、数据问题等异常处理
# 4. **详细的中文注释** - 每个函数和重要步骤都有清晰说明
# 5. **内存优化** - 数据采样、适当的数据类型转换
# 6. **模型评估** - HitRate@3、LogLoss、特征重要性分析
# 7. **大数据集优化** - 分块加载和内存优化
# 
# ### 支持的功能：
# - 数据加载和验证
# - 智能数据采样（避免数据泄露）
# - 多种类型的特征工程
# - XGBoost排序模型训练
# - 全面的模型评估
# - 预测结果生成和保存
# 
# ### 使用方法：
# 只需要运行 `main()` 函数即可完成整个机器学习流程！
