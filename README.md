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

### 確認

```sh
open https://rema424.github.io/remalab/
```

## Google Analytics

### Google Analytics プロパティの作成

Google Analytics アカウントを未所持の場合は作成します。

アカウントが準備できたらプロパティを作成します。

```yml
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

### Google Tag Manager コンテナの作成

Google Tag Manager アカウントを未所持の場合は作成します。

アカウントが準備できたらコンテナを作成します。

```yml
コンテナ名: remalab
ターゲットプラットフォーム: ウェブ
```

コンテナを作成すると `<head>` と `<body>` に設置するスニペットが表示されるので控えておきます。（後で GTM ダッシュボード上で確認することもできます。）

### パーシャルの作成

GTM スニペットを設置するためのファイルを作成していきます。

```sh
# ディレクトリの作成
mkdir -p layouts/partials/gtm/

# ファイル作成
touch layouts/partials/gtm/head.html layouts/partials/gtm/body.html
```

それぞれのファイルに控えておいた GTM スニペットを記載して保存します。

環境変数が `production` の場合のみ、script を出力するようにします。

```html
<!-- layouts/partials/gtm/head.html -->

{{ if eq (getenv "HUGO_ENV") "production" | or (eq .Site.Params.env "production") }}
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','<your-container-id>');</script>
<!-- End Google Tag Manager -->
{{ end }}
```

```html
<!-- layouts/partials/gtm/body.html -->

{{ if eq (getenv "HUGO_ENV") "production" | or (eq .Site.Params.env "production") }}
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=<your-container-id>"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
{{ end }}
```

### パーシャルの読み込み

テーマをカスタマイズしてGTMスニペットを配置していきます。

まずはベースファイルをテーマからプロジェクト側にコピーします。

```sh
# ディレクトリ作成
mkdir layouts/_default/

# ファイルコピー
cp themes/anubis/layouts/_default/baseof.html layouts/_default/baseof.html
```

ファイルがコピーできたら中身を編集します。

```html
<!-- layouts/_default/baseof.html -->

<!DOCTYPE html>
{{ $dataTheme := "" }}
{{ if eq site.Params.style "dark-without-switcher" }}
    {{ $dataTheme = "dark" }}
{{ end }}
<html lang="{{ .Site.LanguageCode }}" data-theme="{{ $dataTheme }}">
<head>
    {{ partial "gtm/head.html" . }} <!-- この行を追加 -->
    {{ block "head" . }}
        {{ partial "head.html" . }}
    {{ end }}
</head>
<body>
    {{ partial "gtm/body.html" . }} <!-- この行を追加 -->
    <a class="skip-main" href="#main">{{ i18n "skipToContent" | humanize }}</a>
    <div class="container">
        <header class="common-header"> 
            {{ block "header" . }}
                {{ partial "header.html" . }}
            {{ end }}
        </header>
        <main id="main" tabindex="-1"> 
            {{ block "main" . }}{{ end }}
        </main>
        {{ block "footer" . }}
            {{ partial "footer.html" . }}
        {{ end }}
    </div>
</body>
</html>
```

### 確認

本番モードでローカルサーバーを起動してアクセスし、開発者ツールでスニペットが配置されているか確認します。

```sh
HUGO_ENV=production hugo server -D
```

### GTM の設定

Google Tag Manager のダッシュボードから変数の設定、タグの作成、公開を行います。

```yaml
Google Tab Manager:
  コンテナ:
    変数:
      ユーザー定義変数:
        - 新規:
            変数名: GA
            変数の設定:
              変数のタイプ: 定数
              値: <Google Analyticsの測定ID>
    タグ:
      新規: 
        - タグ名: Google アナリティクス GA4 設定
          タグの設定:
            タグタイプ: Googleアナリティクス:GA4設定
            測定ID: {{GA}}
          トリガー: All Pages
    公開:
      送信設定: バージョンの公開と作成
      バージョン名: GA4の設定追加
      バージョンの説明: ''
      環境への公開: Live
```

### 確認

本番モードでローカルサーバーを起動してアクセスし、`<head>` 内に gtag があることを確認します。

```
<script type="text/javascript" async="" src="http://www.googletagmanager.com/gtag/js?id=<測定ID>&l=dataLayer&cx=c"></script>
```

次に Google Analytics のダッシュボードを開き、リアルタイムタブで自分のアクセスが計測されていることを確認します。

## Google Search Console

