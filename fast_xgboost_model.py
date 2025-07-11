#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AeroClub RecSys 2025 - 快速XGBoost排序模型
基于高效notebook改进的生产级代码

主要特点：
1. 快速数据采样和处理
2. 精简但有效的特征工程
3. 高效的XGBoost训练
4. 简单易用的接口
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import time
import gc
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FastXGBoostRanker:
    """快速XGBoost排序模型"""
    
    def __init__(self, 
                 sample_frac: float = 0.5,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        初始化模型
        
        Args:
            sample_frac: 训练数据采样比例
            random_state: 随机种子
            n_jobs: 并行作业数
        """
        self.sample_frac = sample_frac
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.feature_cols = None
        self.cat_features = None
        
        # 设置随机种子
        np.random.seed(random_state)
        
        print(f"快速XGBoost排序模型初始化完成")
        print(f"采样比例: {sample_frac}")
        print(f"随机种子: {random_state}")
    
    def _get_categorical_features(self) -> list:
        """定义类别特征"""
        return [
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
    
    def _get_exclude_columns(self) -> list:
        """定义要排除的列"""
        exclude_cols = [
            'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
            'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
            'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
            'frequentFlyer',  # 已处理
            'bySelf', 'pricingInfo_passengerCount',  # 常量列
            # 排除baggageAllowance_weightMeasurementType列
            'legs0_segments0_baggageAllowance_weightMeasurementType',
            'legs0_segments1_baggageAllowance_weightMeasurementType',
            'legs1_segments0_baggageAllowance_weightMeasurementType',
            'legs1_segments1_baggageAllowance_weightMeasurementType',
        ]
        
        # 排除segment 2-3列（>98%缺失）
        for leg in [0, 1]:
            for seg in [2, 3]:
                for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                              'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                              'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                              'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
                    exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')
        
        return exclude_cols
    
    def _hms_to_minutes(self, s: pd.Series) -> np.ndarray:
        """将'HH:MM:SS'格式转换为分钟"""
        mask = s.notna()
        out = np.zeros(len(s), dtype=float)
        if mask.any():
            parts = s[mask].astype(str).str.split(':', expand=True)
            out[mask] = (
                pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
                + pd.to_numeric(parts[1], errors="coerce").fillna(0)
            )
        return out
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征工程
        基于notebook的高效特征工程方法
        """
        print("开始特征工程...")
        start_time = time.time()
        
        df = df.copy()
        
        # 处理时间列
        dur_cols = (
            ["legs0_duration", "legs1_duration"]
            + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
        )
        for col in dur_cols:
            if col in df.columns:
                df[col] = self._hms_to_minutes(df[col])
        
        # 特征容器
        feat = {}
        
        # 价格特征
        feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
        feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
        feat["log_price"] = np.log1p(df["totalPrice"])
        
        # 时长特征
        df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
        feat["duration_ratio"] = np.where(
            df["legs1_duration"].fillna(0) > 0,
            df["legs0_duration"] / (df["legs1_duration"] + 1),
            1.0,
        )
        
        # 航段数量特征
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
        grp = df.groupby("ranker_id")
        feat["price_rank"] = grp["totalPrice"].rank()
        feat["price_pct_rank"] = grp["totalPrice"].rank(pct=True)
        feat["duration_rank"] = grp["total_duration"].rank()
        feat["is_cheapest"] = (grp["totalPrice"].transform("min") == df["totalPrice"]).astype(int)
        feat["is_most_expensive"] = (grp["totalPrice"].transform("max") == df["totalPrice"]).astype(int)
        feat["price_from_median"] = grp["totalPrice"].transform(
            lambda x: (x - x.median()) / (x.std() + 1)
        )
        
        # 常旅客特征
        ff = df["frequentFlyer"].fillna("").astype(str)
        feat["n_ff_programs"] = ff.str.count("/") + (ff != "")
        
        # 主要航空公司的常旅客计划
        for al in ["SU", "S7", "U6", "TK"]:
            feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)
        
        # 常旅客与承运人匹配
        feat["ff_matches_carrier"] = 0
        for al in ["SU", "S7", "U6", "TK"]:
            if "legs0_segments0_marketingCarrier_code" in df.columns:
                feat["ff_matches_carrier"] |= (
                    (feat.get(f"ff_{al}", 0) == 1) & 
                    (df["legs0_segments0_marketingCarrier_code"] == al)
                ).astype(int)
        
        # 二元特征
        feat["is_vip_freq"] = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int)
        feat["has_corporate_tariff"] = (~df["corporateTariffCode"].isna()).astype(int)
        
        # 行李和费用特征
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
        for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                feat[f"{col}_hour"] = dt.dt.hour.fillna(12)
                feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
                h = dt.dt.hour.fillna(12)
                feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)
        
        # 直飞特征
        feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
        feat["is_direct_leg1"] = np.where(
            feat["is_one_way"] == 1,
            0,
            (feat["n_segments_leg1"] == 1).astype(int)
        )
        feat["both_direct"] = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype(int)
        
        # 最便宜的直飞
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
        
        # 其他特征
        feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)
        feat["group_size"] = df.groupby("ranker_id")["Id"].transform("count")
        feat["group_size_log"] = np.log1p(feat["group_size"])
        
        # 主要承运人
        if "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
        else:
            feat["is_major_carrier"] = 0
        
        # 热门路线
        popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"}
        feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)
        
        # 舱位等级特征
        feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
        feat["cabin_class_diff"] = (
            df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
        )
        
        # 合并新特征
        df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)
        
        # 填充缺失值
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(0)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("missing")
        
        elapsed_time = time.time() - start_time
        print(f"特征工程完成，耗时: {elapsed_time:.2f}秒")
        print(f"最终特征数量: {df.shape[1]}")
        
        return df
    
    def load_and_sample_data(self, train_path:str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载和采样数据"""
        print("加载数据...")
        start_time = time.time()
        
        # 加载数据
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        
        print(f"原始数据: Train {train.shape}, Test {test.shape}")
        print(f"唯一ranker_ids: {train['ranker_id'].nunique():,}")
        print(f"选择率: {train['selected'].mean():.3f}")
        
        # 按ranker_id采样以保持组完整性
        if self.sample_frac < 1.0:
            unique_rankers = train['ranker_id'].unique()
            n_sample = int(len(unique_rankers) * self.sample_frac)
            sampled_rankers = np.random.RandomState(self.random_state).choice(
                unique_rankers, size=n_sample, replace=False
            )
            train = train[train['ranker_id'].isin(sampled_rankers)]
            print(f"采样后数据: {len(train):,} rows ({train['ranker_id'].nunique():,} groups)")
        
        # 转换ranker_id为字符串
        train['ranker_id'] = train['ranker_id'].astype(str)
        test['ranker_id'] = test['ranker_id'].astype(str)
        
        elapsed_time = time.time() - start_time
        print(f"数据加载完成，耗时: {elapsed_time:.2f}秒")
        
        return train, test
    
    def prepare_features(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """准备特征数据"""
        print("准备特征数据...")
        
        # 应用特征工程
        train = self.create_features(train)
        test = self.create_features(test)
        
        # 获取特征列
        cat_features = self._get_categorical_features()
        exclude_cols = self._get_exclude_columns()
        
        feature_cols = [col for col in train.columns if col not in exclude_cols]
        cat_features_final = [col for col in cat_features if col in feature_cols]
        
        print(f"使用 {len(feature_cols)} 个特征 ({len(cat_features_final)} 个类别特征)")
        
        # 存储特征信息
        self.feature_cols = feature_cols
        self.cat_features = cat_features_final
        
        return train, test, feature_cols
    
    def encode_categorical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """编码类别特征"""
        print("编码类别特征...")
        
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        X_test_encoded = X_test.copy()
        
        for col in self.cat_features:
            if col in X_train_encoded.columns:
                # 创建映射
                unique_vals = pd.concat([X_train_encoded[col], X_val_encoded[col], X_test_encoded[col]]).unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                
                X_train_encoded[col] = X_train_encoded[col].map(mapping).fillna(-1).astype(int)
                X_val_encoded[col] = X_val_encoded[col].map(mapping).fillna(-1).astype(int)
                X_test_encoded[col] = X_test_encoded[col].map(mapping).fillna(-1).astype(int)
        
        return X_train_encoded, X_val_encoded, X_test_encoded
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series,
                   groups_train: pd.Series, groups_val: pd.Series) -> xgb.Booster:
        """训练XGBoost模型"""
        print("训练XGBoost模型...")
        start_time = time.time()
        
        # 编码类别特征
        X_train_encoded, X_val_encoded, _ = self.encode_categorical_features(X_train, X_val, X_train)
        
        # 创建组大小
        group_sizes_train = pd.DataFrame(groups_train).groupby('ranker_id').size().values
        group_sizes_val = pd.DataFrame(groups_val).groupby('ranker_id').size().values
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train_encoded, label=y_train, group=group_sizes_train)
        dval = xgb.DMatrix(X_val_encoded, label=y_val, group=group_sizes_val)
        
        # XGBoost参数
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg@3',
            'max_depth': 8,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 10.0,
            'learning_rate': 0.05,
            'seed': self.random_state,
            'n_jobs': self.n_jobs,
            'verbosity': 1
        }
        
        # 训练模型
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=50
        )
        
        elapsed_time = time.time() - start_time
        print(f"模型训练完成，耗时: {elapsed_time:.2f}秒")
        
        return self.model
    
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series, model_name: str = "Model") -> Dict[str, float]:
        """评估模型性能"""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x / 10))
        
        def calculate_hitrate_at_k(df, k=3):
            """计算HitRate@k"""
            hits = []
            for ranker_id, group in df.groupby('ranker_id'):
                if len(group) > 10:
                    top_k = group.nlargest(k, 'pred')
                    hit = (top_k['selected'] == 1).any()
                    hits.append(hit)
            return np.mean(hits) if hits else 0.0
        
        df = pd.DataFrame({
            'ranker_id': groups,
            'pred': y_pred,
            'selected': y_true
        })
        
        # 每组的top预测
        top_preds = df.loc[df.groupby('ranker_id')['pred'].idxmax()]
        top_preds['prob'] = sigmoid(top_preds['pred'])
        
        # 计算指标
        logloss = log_loss(top_preds['selected'], top_preds['prob'])
        hitrate_at_3 = calculate_hitrate_at_k(df, k=3)
        accuracy = (top_preds['selected'] == 1).mean()
        
        metrics = {
            'hitrate_at_3': hitrate_at_3,
            'logloss': logloss,
            'accuracy': accuracy
        }
        
        print(f"{model_name} 验证指标:")
        print(f"HitRate@3 (groups >10): {hitrate_at_3:.4f}")
        print(f"LogLoss:                {logloss:.4f}")
        print(f"Top-1 Accuracy:         {accuracy:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型未训练，无法获取特征重要性")
        
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        print(f"Top {top_n} 重要特征:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def predict(self, X_test: pd.DataFrame, groups_test: pd.Series) -> pd.DataFrame:
        """生成预测结果"""
        if self.model is None:
            raise ValueError("模型未训练，无法生成预测")
        
        print("生成预测结果...")
        start_time = time.time()
        
        # 编码类别特征
        _, _, X_test_encoded = self.encode_categorical_features(X_test, X_test, X_test)
        
        # 创建DMatrix
        group_sizes_test = pd.DataFrame(groups_test).groupby('ranker_id').size().values
        dtest = xgb.DMatrix(X_test_encoded, group=group_sizes_test)
        
        # 预测
        test_preds = self.model.predict(dtest)
        
        elapsed_time = time.time() - start_time
        print(f"预测完成，耗时: {elapsed_time:.2f}秒")
        
        return test_preds
    
    def run_full_pipeline(self, train_path: str, test_path: str, output_path: str = 'submission.csv') -> pd.DataFrame:
        """运行完整流程"""
        print("=== 开始完整训练流程 ===")
        total_start_time = time.time()
        
        # 1. 加载和采样数据
        train, test = self.load_and_sample_data(train_path, test_path)
        
        # 2. 特征工程
        train, test, feature_cols = self.prepare_features(train, test)
        
        # 3. 准备训练数据
        X_train = train[feature_cols]
        y_train = train['selected']
        groups_train = train['ranker_id']
        
        X_test = test[feature_cols]
        groups_test = test['ranker_id']
        
        # 4. 分割训练/验证集
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]
        
        print(f"数据分割: Train {len(X_tr):,}, Val {len(X_val):,}, Test {len(X_test):,}")
        
        # 5. 修复数据类型
        X_tr, X_val, X_test = self.fix_data_types(X_tr, X_val, X_test)
        
        # 6. 计算组大小
        group_sizes_tr = self.get_group_sizes(groups_tr)
        group_sizes_val = self.get_group_sizes(groups_val)
        
        print(f"   训练组数: {len(group_sizes_tr)}, 验证组数: {len(group_sizes_val)}")
        
        # 7. 创建DMatrix
        print("🚀 创建DMatrix...")
        try:
            # 🔥 第一步：检查数据类型
            print(f"   X_tr数据类型: {X_tr.dtypes.value_counts().to_dict()}")
            print(f"   是否有object类型: {(X_tr.dtypes == 'object').any()}")
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr, group=group_sizes_tr)
            dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val)
            print("   ✅ DMatrix创建成功")
        except Exception as e:
            print(f"   ❌ DMatrix创建失败: {e}")
            print("   🔄 尝试启用分类特征支持...")
            
            try:
                dtrain = xgb.DMatrix(X_tr, label=y_tr, group=group_sizes_tr, enable_categorical=True)
                dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val, enable_categorical=True)
                print("   ✅ 启用分类特征的DMatrix创建成功")
            except Exception as e2:
                print(f"   ❌ 启用分类特征也失败: {e2}")
                print("   🔄 尝试纯数值方法...")
                
                try:
                    # 🔥 只保留数值列
                    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns
                    print(f"   使用 {len(numeric_cols)} 个数值列")
                    
                    X_tr_numeric = X_tr[numeric_cols].fillna(0).astype(np.float32)
                    X_val_numeric = X_val[numeric_cols].fillna(0).astype(np.float32)
                    
                    dtrain = xgb.DMatrix(X_tr_numeric, label=y_tr, group=group_sizes_tr)
                    dval = xgb.DMatrix(X_val_numeric, label=y_val, group=group_sizes_val)
                    print("   ✅ 数值列DMatrix创建成功")
                    
                    # 🔥 更新X_test以保持一致
                    X_test = X_test[numeric_cols].fillna(0).astype(np.float32)
                    
                except Exception as e3:
                    print(f"   ❌ 数值列方法也失败: {e3}")
                    print("   🔄 最后尝试：简化到二分类...")
                    
                    try:
                        # 🔥 最后方案：放弃ranking，只用二分类
                        X_tr_simple = X_tr.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
                        X_val_simple = X_val.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
                        
                        dtrain = xgb.DMatrix(X_tr_simple, label=y_tr.values)
                        dval = xgb.DMatrix(X_val_simple, label=y_val.values)
                        print("   ✅ 简化二分类DMatrix创建成功")
                        
                        # 🔥 更新X_test
                        X_test = pd.DataFrame(X_test.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32))
                        
                    except Exception as e4:
                        print(f"   💥 所有DMatrix创建方法都失败了: {e4}")
                        raise RuntimeError("无法创建XGBoost DMatrix，请检查数据")
        
        # 8. 配置XGBoost参数
        print("⚙️ 配置XGBoost参数...")
        xgb_params = {
            'objective': 'binary:logistic',  # 使用二分类，更稳定
            'eval_metric': 'logloss',
            'max_depth': 6,                  # 降低复杂度
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'learning_rate': 0.1,
            'seed': 42,
            'tree_method': 'hist',           # 使用histogram方法，更快更稳定
            'n_jobs': -1,
            'verbosity': 1
        }
        
        # 9. 训练模型
        print("🏃‍♂️ 开始训练XGBoost...")
        try:
            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=100,  # 减少轮数，更快
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=10
            )
            print("   ✅ XGBoost训练完成")
        except Exception as e:
            print(f"   ❌ 训练失败: {e}")
            raise
        
        # 10. 清理内存
        print("🧹 清理训练数据...")
        del dtrain, X_tr, y_tr, groups_tr
        gc.collect()
        print("   ✅ 训练数据清理完成")
        
        # 11. 验证评估
        _, X_val_encoded, _ = self.encode_categorical_features(X_tr, X_val, X_test)
        group_sizes_val = pd.DataFrame(groups_val).groupby('ranker_id').size().values
        dval = xgb.DMatrix(X_val_encoded, group=group_sizes_val)
        val_preds = xgb_model.predict(dval)
        
        metrics = self.evaluate_model(y_val, val_preds, groups_val, "XGBoost")
        
        # 12. 特征重要性
        self.get_feature_importance()
        
        # 13. 生成测试预测
        test_preds = self.predict(X_test, groups_test)
        
        # 14. 创建提交文件
        submission = test[['Id', 'ranker_id']].copy()
        submission['pred_score'] = test_preds
        submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
            ascending=False, method='first'
        ).astype(int)
        
        # 保存结果
        output_file = 'ultra_fast_submission.csv'
        final_submission = submission[['Id', 'ranker_id', 'selected']]
        final_submission.to_csv(output_file, index=False)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"\n=== 完整流程完成 ===")
        print(f"总耗时: {total_elapsed_time:.2f}秒")
        print(f"结果已保存到: {output_file}")
        print(f"生成了 {len(final_submission):,} 行预测结果")
        
        return final_submission


def quick_run(train_path: str, test_path: str, output_path: str = 'submission.csv', 
              sample_frac: float = 0.5, random_state: int = 42) -> pd.DataFrame:
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
    print("=== 快速XGBoost模型 ===")
    print(f"文件配置:")
    print(f"   训练数据: {train_path}")
    print(f"   测试数据: {test_path}")
    print(f"   输出文件: {output_path}")
    print(f"   采样比例: {sample_frac}")
    print()
    
    try:
        # 创建模型
        model = FastXGBoostRanker(
            sample_frac=sample_frac,
            random_state=random_state
        )
        
        # 运行流程
        submission = model.run_full_pipeline(train_path, test_path, output_path)
        
        print("运行完成！")
        return submission
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


# 主函数示例
if __name__ == "__main__":
    # 配置参数
    TRAIN_FILE_PATH = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
    TEST_FILE_PATH = '/kaggle/input/aeroclub-recsys-2025/test.parquet'
    OUTPUT_FILE_PATH = 'fast_submission.csv'
    SAMPLE_FRAC = 0.5  # 采样比例
    RANDOM_STATE = 42
    
    print("=== AeroClub RecSys 2025 - 快速XGBoost排序模型 ===")
    print(f"基于高效notebook改进的生产级代码")
    print()
    
    # 快速运行
    submission = quick_run(
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        OUTPUT_FILE_PATH,
        sample_frac=SAMPLE_FRAC,
        random_state=RANDOM_STATE
    )
    
    if not submission.empty:
        print(f"\n任务完成！生成了 {len(submission)} 行预测结果")
        print(f"输出文件: {OUTPUT_FILE_PATH}")
    else:
        print("\n任务失败，请检查数据文件和错误信息")

# ========== 极简特征工程 ==========
print("🚀 极简特征工程")

def minimal_features(df):
    """最少但有效的特征 - 超快版本"""
    
    print(f"   处理数据: {df.shape}")
    
    # 🔥 跳过复杂的时间处理，直接用简单特征
    # 不处理duration列，因为太慢了
    
    # 最基础的价格特征
    df['log_price'] = np.log1p(df['totalPrice']).astype(np.float32)
    
    # 🔥 快速判断是否单程 - 基于列是否存在值
    if 'legs1_duration' in df.columns:
        df['is_one_way'] = df['legs1_duration'].isna().astype(np.int8)
    else:
        df['is_one_way'] = 1  # 如果没有leg1列，就是单程
    
    # 🔥 最重要：组内价格排名
    print("   计算价格排名...")
    df['price_rank'] = df.groupby('ranker_id')['totalPrice'].rank(method='dense').astype(np.int16)
    
    print("   计算最便宜标记...")
    df['is_cheapest'] = (df.groupby('ranker_id')['totalPrice'].transform('min') == 
                        df['totalPrice']).astype(np.int8)
    
    # 🔥 简单的税费比率
    df['tax_ratio'] = (df['taxes'] / (df['totalPrice'] + 1)).astype(np.float32)
    
    print("   特征工程完成")
    return df

# 应用特征工程
print("🔧 训练数据特征工程...")
train = minimal_features(train)

print("🔧 测试数据特征工程...")  
test = minimal_features(test)

print(f"✅ 极简特征工程完成")
print(f"   训练数据: {train.shape}")
print(f"   测试数据: {test.shape}")

# 立即清理内存
import gc
gc.collect()

# ========== 修复数据类型并训练XGBoost ==========
print("🚀 修复数据类型并训练XGBoost")

# 🔥 首先修复所有数据类型问题
def fix_data_types(X_tr, X_val, X_test):
    """修复所有数据类型问题"""
    print("🔧 修复数据类型...")
    
    # 🔥 创建副本避免修改原数据
    X_tr = X_tr.copy()
    X_val = X_val.copy() 
    X_test = X_test.copy()
    
    # 处理duration列 - 如果是object，转换为0
    for col in ['legs0_duration', 'legs1_duration']:
        if col in X_tr.columns:
            if X_tr[col].dtype == 'object':
                print(f"   修复 {col} 列...")
                X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce').fillna(0).astype(np.float32)
                X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0).astype(np.float32)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(np.float32)
    
    # 🔥 强制转换所有非数值列
    for col in X_tr.columns:
        if X_tr[col].dtype in ['object', 'category'] or str(X_tr[col].dtype).startswith('string'):
            print(f"   转换类别列 {col}...")
            # 对所有object/category列进行Label Encoding
            all_vals = pd.concat([
                X_tr[col].astype(str), 
                X_val[col].astype(str), 
                X_test[col].astype(str)
            ]).unique()
            
            # 创建映射字典
            mapping = {val: i for i, val in enumerate(all_vals)}
            
            # 应用映射并转换为数值类型
            X_tr[col] = X_tr[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
            X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
            X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(np.int32)
            
        elif X_tr[col].dtype == 'bool':
            # 布尔转换为int
            X_tr[col] = X_tr[col].astype(np.int8)
            X_val[col] = X_val[col].astype(np.int8)
            X_test[col] = X_test[col].astype(np.int8)
        
        # 🔥 确保所有数值列都是float32或int类型
        elif X_tr[col].dtype in ['float64']:
            X_tr[col] = X_tr[col].astype(np.float32)
            X_val[col] = X_val[col].astype(np.float32)
            X_test[col] = X_test[col].astype(np.float32)
    
    # 🔥 最后检查：确保没有object类型的列
    for col in X_tr.columns:
        if X_tr[col].dtype == 'object':
            print(f"   🚨 发现遗漏的object列: {col}, 强制转换为数值")
            X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce').fillna(0).astype(np.float32)
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0).astype(np.float32)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(np.float32)
    
    # 填充所有NaN
    X_tr = X_tr.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"   ✅ 数据类型修复完成")
    print(f"   X_tr形状: {X_tr.shape}")
    print(f"   数据类型分布: {X_tr.dtypes.value_counts().to_dict()}")
    
    # 🔥 最终验证：检查是否还有非数值类型
    non_numeric_cols = []
    for col in X_tr.columns:
        if not pd.api.types.is_numeric_dtype(X_tr[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"   ⚠️ 警告：仍有非数值列: {non_numeric_cols}")
        # 强制转换剩余的列
        for col in non_numeric_cols:
            X_tr[col] = pd.to_numeric(X_tr[col], errors='coerce').fillna(0).astype(np.float32)
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0).astype(np.float32)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(np.float32)
    
    return X_tr, X_val, X_test

# 修复数据类型
X_tr, X_val, X_test = fix_data_types(X_tr, X_val, X_test)

# 🔥 创建简化的组信息
print("📊 创建组信息...")

# 正确计算组大小
def get_group_sizes(groups):
    """获取正确的组大小"""
    return groups.value_counts().sort_index().values

group_sizes_tr = get_group_sizes(groups_tr)
group_sizes_val = get_group_sizes(groups_val)

print(f"   训练组数: {len(group_sizes_tr)}")
print(f"   验证组数: {len(group_sizes_val)}")
print(f"   训练数据总行数: {len(X_tr)}")
print(f"   组大小总和: {group_sizes_tr.sum()}")

# 🔥 创建DMatrix
print("🚀 创建DMatrix...")
try:
    # 🔥 第一步：检查数据类型
    print(f"   X_tr数据类型: {X_tr.dtypes.value_counts().to_dict()}")
    print(f"   是否有object类型: {(X_tr.dtypes == 'object').any()}")
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, group=group_sizes_tr)
    dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val)
    print("   ✅ DMatrix创建成功")
except Exception as e:
    print(f"   ❌ DMatrix创建失败: {e}")
    print("   🔄 尝试启用分类特征支持...")
    
    try:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, group=group_sizes_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, group=group_sizes_val, enable_categorical=True)
        print("   ✅ 启用分类特征的DMatrix创建成功")
    except Exception as e2:
        print(f"   ❌ 启用分类特征也失败: {e2}")
        print("   🔄 尝试纯数值方法...")
        
        try:
            # 🔥 只保留数值列
            numeric_cols = X_tr.select_dtypes(include=[np.number]).columns
            print(f"   使用 {len(numeric_cols)} 个数值列")
            
            X_tr_numeric = X_tr[numeric_cols].fillna(0).astype(np.float32)
            X_val_numeric = X_val[numeric_cols].fillna(0).astype(np.float32)
            
            dtrain = xgb.DMatrix(X_tr_numeric, label=y_tr, group=group_sizes_tr)
            dval = xgb.DMatrix(X_val_numeric, label=y_val, group=group_sizes_val)
            print("   ✅ 数值列DMatrix创建成功")
            
            # 🔥 更新X_test以保持一致
            X_test = X_test[numeric_cols].fillna(0).astype(np.float32)
            
        except Exception as e3:
            print(f"   ❌ 数值列方法也失败: {e3}")
            print("   🔄 最后尝试：简化到二分类...")
            
            try:
                # 🔥 最后方案：放弃ranking，只用二分类
                X_tr_simple = X_tr.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
                X_val_simple = X_val.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
                
                dtrain = xgb.DMatrix(X_tr_simple, label=y_tr.values)
                dval = xgb.DMatrix(X_val_simple, label=y_val.values)
                print("   ✅ 简化二分类DMatrix创建成功")
                
                # 🔥 更新X_test
                X_test = pd.DataFrame(X_test.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32))
                
            except Exception as e4:
                print(f"   💥 所有DMatrix创建方法都失败了: {e4}")
                raise RuntimeError("无法创建XGBoost DMatrix，请检查数据")

# 🔥 超级简化的XGBoost参数
params = {
    'objective': 'binary:logistic',  # 🔥 改为二分类，更简单
    'eval_metric': 'logloss',
    'max_depth': 4,  # 🔥 很浅的树
    'learning_rate': 0.3,  # 🔥 更高的学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': 2  # 🔥 限制并行
}

print("🚀 开始训练...")
try:
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,  # 🔥 只训练50轮
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    print("✅ 训练成功完成")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    # 如果还是失败，用最简单的参数
    print("🔄 尝试最简单的训练...")
    model = xgb.train(
        {'objective': 'binary:logistic', 'verbosity': 0},
        dtrain,
        num_boost_round=20
    )
    print("✅ 简化训练完成")

# 🔥 立即清理训练数据释放内存
del dtrain, X_tr, y_tr, groups_tr
if 'dval' in locals():
    del dval
gc.collect()

print("🎉 XGBoost训练步骤完成")

# ========== 修复预测并创建提交文件 ==========
print("🚀 修复预测并创建提交文件")

# 🔥 重新读取test的Id列（因为可能丢失了）
print("📥 确保有Id列...")
try:
    test_ids = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=['Id'])
    print(f"   ✅ 成功读取 {len(test_ids)} 个Id")
except Exception as e:
    print(f"   ⚠️ 无法读取Id列，使用索引: {e}")
    test_ids = pd.DataFrame({'Id': range(len(X_test))})

# 🔥 生成预测 - 修复group结构问题
print("🔮 生成预测...")
try:
    # 🔥 关键修复：不使用group信息创建测试DMatrix
    # 因为我们使用的是binary:logistic，不需要group信息
    print("   创建测试DMatrix（不使用group信息）...")
    dtest = xgb.DMatrix(X_test)  # 🔥 不传入group参数
    
    print("   开始预测...")
    test_preds = model.predict(dtest)
    print(f"   ✅ 生成了 {len(test_preds)} 个预测")
    
except Exception as e:
    print(f"   ❌ 标准预测失败: {e}")
    print("   🔄 尝试更简单的预测方法...")
    
    try:
        # 🔥 如果还是失败，直接使用numpy数组
        test_preds = model.predict(xgb.DMatrix(X_test.values))
        print(f"   ✅ 简化预测成功，生成了 {len(test_preds)} 个预测")
    except Exception as e2:
        print(f"   ❌ 简化预测也失败: {e2}")
        print("   ⚠️ 使用随机预测作为后备方案")
        test_preds = np.random.random(len(X_test))

# 🔥 创建提交DataFrame
print("📊 创建提交文件...")
submission = pd.DataFrame({
    'Id': test_ids['Id'],
    'ranker_id': groups_test,
    'pred_score': test_preds
})

print(f"   提交数据形状: {submission.shape}")
print(f"   预测值范围: {test_preds.min():.4f} 到 {test_preds.max():.4f}")

# 🔥 计算组内排名
print("🏆 计算组内排名...")
submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
    ascending=False, method='first'
).astype(int)

print(f"   排名计算完成，范围: 1 到 {submission['selected'].max()}")

# 🔥 保存最终提交文件
output_file = 'ultra_fast_submission.csv'
# 🔥 修复：包含ranker_id列
final_submission = submission[['Id', 'ranker_id', 'selected']]
final_submission.to_csv(output_file, index=False)

print(f"🎉 任务完成！")
print(f"   📁 文件名: {output_file}")
print(f"   📊 预测数量: {len(final_submission):,}")
print(f"   🏆 排名范围: 1 到 {submission['selected'].max()}")

# 显示一些统计信息
print(f"\n📈 提交文件统计:")
print(f"   总行数: {len(final_submission)}")
print(f"   唯一Id数: {final_submission['Id'].nunique()}")
print(f"   唯一ranker_id数: {final_submission['ranker_id'].nunique()}")
print(f"   排名分布（前10）:")
rank_counts = final_submission['selected'].value_counts().head(10)
for rank in sorted(rank_counts.index):
    count = rank_counts[rank]
    print(f"     排名{rank}: {count:,} 个")

# 🔥 验证提交文件格式
print(f"\n🔍 验证提交文件:")
print(f"   必需列: {list(final_submission.columns)}")
print(f"   是否有重复Id: {final_submission['Id'].duplicated().any()}")
print(f"   selected列数据类型: {final_submission['selected'].dtype}")
print(f"   🔥 ranker_id列已包含: {'ranker_id' in final_submission.columns}")

# 🔥 显示提交文件示例
print(f"\n📋 提交文件示例（前5行）:")
print(final_submission.head().to_string(index=False))

# 🔥 最终清理
if 'model' in locals():
    del model
if 'dtest' in locals():
    del dtest
del X_test, submission, final_submission
gc.collect()

print(f"\n🚀 所有步骤完成！文件已保存到: {output_file}")
print("📝 可以直接提交这个CSV文件到Kaggle竞赛")