import json
import logging
import random
from datetime import datetime, timedelta
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import polars as pl

from src.consts import INSTANCE_PATH
from src.utils import set_logger

logger: Logger = logging.getLogger(__name__)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 score without scikit-learn."""
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def make_predicotr(
    df: pl.DataFrame, feature_cols: list[str], target_col: str, n_estimators: int
) -> lgb.Booster:
    """
    予測モデルを作成する
    Args:
        df: データフレーム
        feature_cols: 特徴量の列名
        target_col: 目的変数の列名
        n_estimators: 木の本数
    Returns:
        予測モデル
    """
    logger.info(f"Making predictor with {n_estimators} estimators")
    # Polars -> NumPy
    X = df.select(feature_cols).to_numpy()
    y = df.select(target_col).to_series().to_numpy()

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "verbosity": -1,  # -1: 警告やエラーメッセージを出力しない
    }
    booster = lgb.train(params, train_data, num_boost_round=n_estimators)

    y_pred = booster.predict(X)
    logger.info(f"R2 score: {_r2_score(y, y_pred)}")
    return booster


def generate_units(n_units: int, base_date: str = "2023-08-06") -> List[Dict[str, Any]]:
    """
    週次でユニットを生成する
    Args:
        n_units: 生成するユニット数
        base_date: 基準日 (YYYY-MM-DD)
    Returns:
        ユニットリスト
    """
    base = datetime.strptime(base_date, "%Y-%m-%d")
    units = []

    for i in range(n_units):
        # 週次で日付を進める
        date = base + timedelta(weeks=i)
        units.append(
            {
                "unit_id": i,
                "constant_features": {
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                },
            }
        )

    return units


def get_feature_bounds(
    df: pl.DataFrame, opt_features: List[str]
) -> List[Dict[str, Any]]:
    """
    各決定変数の範囲を[0, 500]に設定する
    Args:
        df: データフレーム
        opt_features: 最適化対象特徴量
    Returns:
        特徴量範囲の設定
    """
    bounds = []
    for feature in opt_features:
        bounds.append(
            {
                "name": feature,
                "lower_bound": 0.0,
                "upper_bound": 500.0,
                "type": "continuous",
            }
        )

    return bounds


def generate_sum_constraints(
    n_units: int, opt_features: List[str]
) -> List[Dict[str, Any]]:
    """
    各ユニットごとに x1+x2+...+x7 の総和が1000以下の制約を生成する
    Args:
        n_units: ユニット数
        opt_features: 最適化対象特徴量
    Returns:
        制約条件リスト
    """
    constraints = []

    # 各ユニットについて全特徴量の総和が1000以下の制約
    for unit_id in range(n_units):
        variables = []
        for feature in opt_features:
            variables.append(
                {"unit_id": unit_id, "feature": feature, "coefficient": 1.0}
            )

        constraints.append(
            {
                "constraint_id": f"unit_{unit_id}_total",
                "description": f"Total sum constraint for unit {unit_id}",
                "variables": variables,
                "upper_bound": 1000.0,
            }
        )

    return constraints


def create_instance_json(
    n_units: int,
    n_estimators: int,
    opt_features: List[str],
    const_features: List[str],
    df: pl.DataFrame,
    model_filename: str,
) -> Dict[str, Any]:
    """
    インスタンスのJSON構造を生成する
    """
    units = generate_units(n_units)
    feature_bounds = get_feature_bounds(df, opt_features)
    sum_constraints = generate_sum_constraints(n_units, opt_features)

    instance = {
        "n_units": n_units,
        "n_optimization_features": len(opt_features),
        "n_constant_features": len(const_features),
        "n_sum_constraints": len(sum_constraints),
        "features": {
            "optimization_features": feature_bounds,
            "constant_features": const_features,
        },
        "units": units,
        "constraints": {"sum_constraints": sum_constraints},
        "model": {
            "file_path": model_filename,
            "input_features_order": opt_features + const_features,
        },
    }

    return instance


def main():
    # ロガーの設定
    set_logger()

    random.seed(42)

    output_dir = INSTANCE_PATH / "ml-opt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    raw_data_path = Path(__file__).parent / "data" / "data.csv"
    df = pl.read_csv(raw_data_path)
    opt_feature_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
    const_feature_cols = ["year", "month", "day"]
    feature_cols = opt_feature_cols + const_feature_cols
    target_col = "y"

    n_estimators_list = [100, 500, 1000, 5000]
    n_units_list = [1, 10, 50]

    # 学習済みモデルを作成
    for n_estimators in n_estimators_list:
        booster = make_predicotr(
            df, feature_cols, target_col, n_estimators=n_estimators
        )
        # モデルファイルを保存
        model_path = output_dir / f"lgbm_{n_estimators}.txt"
        booster.save_model(str(model_path))

        # 各ユニット数でJSONインスタンスを生成
        for n_units in n_units_list:
            logger.info(
                f"Generating instance: n_units={n_units}, n_estimators={n_estimators}"
            )

            instance = create_instance_json(
                n_units=n_units,
                n_estimators=n_estimators,
                opt_features=opt_feature_cols,
                const_features=const_feature_cols,
                df=df,
                model_filename=f"lgbm_{n_estimators}.txt",
            )

            # JSONファイルを保存
            json_path = output_dir / f"instance_{n_units}units_{n_estimators}est.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(instance, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved instance to {json_path}")


if __name__ == "__main__":
    main()
