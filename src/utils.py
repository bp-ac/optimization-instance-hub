import logging


def set_logger(level: int = logging.INFO) -> None:
    """
    ロガーの基本設定を行う
    Args:
        level: ログレベル (デフォルト: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
