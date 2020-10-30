# RemaLab

## Getting Started

### インストール

```sh
# インストール
brew install hugo

# バージョン確認
hugo version

# プロジェクト作成
hugo new site <your-project-name>

# ディレクトリ移動
cd <your-project-name>

# git 初期化
git init

# テーマの追加（今回は Anubis を使用）
git submodule add https://github.com/mitrichius/hugo-theme-anubis.git themes/anubis

# 設定ファイルにテーマを追加
echo 'theme = "anubis"' >> config.toml

# 記事ファイルの作成
hugo new <category>/<article-title>.md

# コミット

git add .

git commit -m "first post"

# サーバー起動（ドラフトも表示）
hugo server -D

# ブラウザで表示
open http://localhost:1313
```

### 設定の編集

`config.toml` を編集

```toml
baseURL = "http://example.org/"
languageCode = "ja-jp"
title = "RemaLab"
theme = "anubis"
```

### ビルド

```sh
hugo -D
```

## Host on GitHub

GitHub で新規プロジェクトを作成します（今回は remalab で作成）。

作成できたら Git リポジトリを push します。

```sh
git remote add origin https://github.com/rema424/remalab.git

git branch -M main

git push -u origin main
```

`config.toml` を編集する。

```toml
baseURL = "https://rema424.github.io/remalab/"
publishDir = "docs"
```

既存のビルド先ディレクトリを削除する。

```sh
rm -r public
```

ビルド（テストのためドラフトも含む）。

```sh
hugo -D
```

ビルド済みソースコードを push する。

```sh
git add .

git commit -m "first deploy"

git push
```

GitHub の設定を編集する。

GitHub > Project > Settings > GitHub Pages

```yaml
Source:
  branch: main
  folder: /root
```

ブラウザで表示する。

```sh
open https://rema424.github.io/remalab/
```
