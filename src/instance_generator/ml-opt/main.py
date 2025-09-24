import logging
import random
from logging import Logger
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.consts import INSTANCE_PATH

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
        # 必要なら他のハイパーパラメータをここに追加
    }
    booster = lgb.train(params, train_data, num_boost_round=n_estimators)

    y_pred = booster.predict(X)
    logger.info(f"R2 score: {_r2_score(y, y_pred)}")
    return booster


def main():
    random.seed(42)

    output_dir = INSTANCE_PATH / "ml-opt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    raw_data_path = Path(__file__).parent / "data" / "data.csv"
    df = pl.read_csv(raw_data_path)
    feature_cols = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
    ]
    target_col = "y"

    n_estimators_list = [100, 500, 1000, 5000]

    for n_estimators in n_estimators_list:
        booster = make_predicotr(
            df, feature_cols, target_col, n_estimators=n_estimators
        )
        # 出力先: instances/ml-opt 配下
        save_path = output_dir / f"lgbm_{n_estimators}.txt"
        booster.save_model(str(save_path))


if __name__ == "__main__":
    main()
