# Optimization Instance Hub

最適化問題のインスタンスを公開・共有するためのリポジトリです。  
実問題に近いベンチマークインスタンスを提供することを目的としています。


## 現在提供している問題

| 問題種別 | 説明 | インスタンス数 | 形式 |
|---------|------|---------------|------|
| **Bin Packing** | ビンパッキング問題 | 81個 | JSON |
| **ML-Opt** | 機械学習モデルを用いた最適化問題 | 複数 | TXT/JSON |


## プロジェクト構造

```
optimization-instance-hub/
├── docs/                          # ドキュメント・サイト生成関連
│   ├── mkdocs.yml                 # MkDocs設定ファイル
│   ├── scripts/                   # インスタンス一覧自動生成スクリプト
│   ├── docs/                      # サイト用アセット（JS・CSS）
│   ├── site/                      # ビルド済み静的サイト（GitHub Pages用）
│   ├── index_header.md            # インデックスページのヘッダー
│   └── index_footer.md            # インデックスページのフッター
│
├── instances/                     # 最適化問題インスタンス群
│   ├── ml-opt/                   # 機械学習予測最適化問題
│   │   ├── (インスタンスファイル群)
│   │   └── docs/                 # 問題説明・図表等
│   …
│
└── src/                          # インスタンス生成・ユーティリティ
    ├── consts.py                # パス定数の定義
    ├── utils.py                 # ロガー設定等のユーティリティ
    └── instance_generator/      # 問題種別ごとのインスタンス生成スクリプト
```


## インスタンスの追加方法

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/bp-ac/optimization-instance-hub.git
cd optimization-instance-hub

# 依存関係をインストール
uv sync
```

### ローカルでのサイト確認

```bash
# ドキュメントサイトをローカルで起動
cd docs
uv run mkdocs serve
```

### 新しい問題インスタンスの追加

1. `instances/` 下に問題種別のディレクトリを作成
2. `instances/instance_name/docs/description.md` でインスタンスの説明を記述
3. `instances/instance_name/`配下にインスタンスファイルを配置
4. トップページにインスタンス説明ページがが自動的に生成されます
