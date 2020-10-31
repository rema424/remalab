# RemaLab

## Getting Started

### インストール

```sh
# インストール
brew install hugo

# バージョン確認
hugo version
```

### プロジェクト作成

```sh
hugo new site <your-project-name>
```

### テーマの設定

```sh
# ディレクトリ移動
cd <your-project-name>

# git 初期化
git init

# テーマの追加（今回は Anubis を使用）
git submodule add https://github.com/mitrichius/hugo-theme-anubis.git themes/anubis

# 設定ファイルにテーマを追加
echo 'theme = "anubis"' >> config.toml
```

### 記事の作成

```sh
hugo new <category>/<article-title>.md
# => content/<catogory>/<argicle-title>.md が生成される
```

### ローカルサーバーの起動

```sh
# サーバー起動（ドラフトも表示）
hugo server -D

# ブラウザで表示
open http://localhost:1313
```

### 設定ファイルの編集

```toml
# config.toml
baseURL = "http://example.org/"
languageCode = "ja-jp"
title = "RemaLab"
theme = "anubis"
```

### ビルド

```sh
# ドラフトも含める場合
hugo -D

# ドラフトを含めない場合
hugo
```

## Host on GitHub

### リモートリポジトリの作成

GitHub で新規プロジェクトを作成します（今回は remalab で作成）。

リモートリポジトリが作成できたら push します。

```sh
# 変更をステージ
git add .

# コミット
git commit -m "first commit"

# リモートリポジトリの追加
git remote add origin https://github.com/rema424/remalab.git

# ブランチの切替
git branch -M main

# puch
git push -u origin main
```

### 設定ファイルの編集

GitHub Pages には **User/Organization Pages** と **Project Pages** の 2 タイプがあります。

今回は **Project Pages** を採用します。(`https://<username>.github.io/<project>/`)

```toml
# config.toml
baseURL = "https://rema424.github.io/remalab/"
publishDir = "docs"
```

### GitHub の設定変更

GitHub > Project > Settings > GitHub Pages より設定を変更します。

```yaml
Source:
  branch: main
  folder: /docs
```

### ローカルサーバーの起動

設定を変更したため、一度ローカルで確認します。

```sh
# サーバー起動（ドラフトも表示）
hugo server -D
# => 設定変更により TOP が localhost:1313/ から localhost:1313/remalab/ へ

# ブラウザで表示
open http://localhost:1313/remalab/
```

### 再ビルド

```sh
# 既存のビルド先ディレクトリを削除
rm -r public

# ビルド（テストのためドラフトも含む）
hugo -D
```

### デプロイ

```sh
git add .

git commit -m "first deploy"

git push
```

GitHub の設定を編集する。

GitHub > Project > Settings > GitHub Pages

ブラウザで表示する。

```sh
open https://rema424.github.io/remalab/
```

## Google Analytics

Google Analytics の プロパティを作成する。

```yaml
プロパティ名: remalab
レポートのタイムゾーン: 日本
通貨: 日本円
業種: Computers & Electronics
ビジネスの規模: 小規模
利用目的: サイトまたはアプリでの顧客エンゲージメントを測定する
Choose a platform: ウェブ
ウェブサイトのURL: https://rema424.github.io/remalab/
ストリーム名: RemaLab
```
