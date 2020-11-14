---
title: 【2020年11月版】PyTorchとBERTで文章の分散表現（ベクトル）を取得する
description: PyTorchでBERTを利用して文章の分散表現（ベクトル）を取得する方法を確認します
date: 2020-11-13T21:13:05+09:00
draft: false
---

日本語における自然言語処理のライブラリ整備がここ1年で大きく進んだみたいです。
これに伴って情報の古い記事も多くなったように感じられたので記しておきます。

以前はBERTで日本語を扱う場合は形態素解析やBERTトークン化を自身で行う必要があったようですが、今はライブラリに一任できます。環境構築も非常に簡単でした。

## setup

```sh
# python3.8のインストール（2020年11月現在、3.9だとエラーになるライブラリが多いため3.8を利用する）
brew install python@3.8

# 仮想環境の作成
python3 -m venv venv

# 仮想環境をアクティベート
. venv/bin/activate

# pipをアップグレード
python -m pip install --upgrade pip

# pytorch（機械学習ライブラリ）のインストール
pip install torch

# transformers（自然言語処理における学習済みモデル提供ライブラリ）のインストール
pip install transformers

# numpy（行列計算ライブラリ）のインストール
pip install numpy

# fugashi（形態素解析器）のインストール
pip install fugashi

# ipadic（形態素解析用辞書）のインストール
pip install ipadic

# mojimoji（日本語文字列の半角・全角変換ライブラリ）のインストール
pip install mojimoji
```

## サンプルコード

```py
import torch
from transformers import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
import re
import mojimoji
import numpy as np
from pprint import pprint as p


# GPUが利用できるか確認
p(torch.cuda.is_available())

# GPUとCPUのどちらを利用するか判定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p(device)

# トークナイザーの読み込み
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

# 学習済みモデルの読み込み
model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")


def preprocess(text):
    """
    日本語文字列の前処理を行う
    """
    # 全角文字を半角に変換
    text = mojimoji.zen_to_han(text)

    # 空白と改行を削除し、コンマとピリオドを句点と読点に変換
    text = (
        text.replace(" ", "")
        .replace("\n", "")
        .replace(",", "、")
        .replace(".", "。")
        .strip()
    )

    # 数字を#に変換
    while bool(re.search(r"[0-9]", text)):
        text = re.sub("[0-9]{5,}", "#####", text)
        text = re.sub("[0-9]{4}", "####", text)
        text = re.sub("[0-9]{3}", "###", text)
        text = re.sub("[0-9]{2}", "##", text)
        text = re.sub("[0-9]{1}", "#", text)

    # 半角文字を全角に変換
    return mojimoji.han_to_zen(text)


def split_to_sentences(text):
    """
    文章を文単位で分割しリストで返却する
    """
    text = re.sub("。", "。\n", text)
    return [s for s in text.split("\n") if s != ""]


def aggregate_tensor(embeddings, pooling_strategy="REDUCE_MEAN_MAX"):
    """
    文ベクトルを集計して文章ベクトルを返却する
    """
    if pooling_strategy == "REDUCE_MEAN":
        return torch.mean(embeddings, dim=0)
    elif pooling_strategy == "REDUCE_MAX":
        max, _ = torch.max(embeddings, dim=0)
        return max
    elif pooling_strategy == "REDUCE_MEAN_MAX":
        max, _ = torch.max(embeddings, dim=0)
        mean = torch.mean(embeddings, dim=0)
        return torch.hstack((max, mean))


def aggregate_numpy(embeddings, pooling_strategy="REDUCE_MEAN_MAX"):
    """
    文ベクトルを集計して文章ベクトルを返却する
    """
    if pooling_strategy == "REDUCE_MEAN":
        return np.mean(embeddings, axis=0)
    elif pooling_strategy == "REDUCE_MAX":
        return np.max(embeddings, axis=0)
    elif pooling_strategy == "REDUCE_MEAN_MAX":
        return np.r_[np.max(embeddings, axis=0), np.mean(embeddings, axis=0)]


def embedding(text):
    # 前処理
    text = preprocess(text)

    # 文単位に分割
    sentences = split_to_sentences(text)

    # BERTトークン化
    encoded = tokenizer.batch_encode_plus(
        sentences, padding=True, add_special_tokens=True
    )

    # BERTトークンID列を抽出
    input_ids = torch.tensor(encoded["input_ids"], device=device)

    # BERTの最大許容トークン数が512なので超える場合は切り詰める
    input_ids = input_ids[:, :512]

    with torch.no_grad():  # 勾配計算なし
        # 単語ベクトルを計算
        outputs = model(input_ids)

    # 最終層の隠れ状態ベクトルを取得
    last_hidden_states = outputs[0]

    # 各文における[CLS]トークンの単語ベクトルを抽出（これを文ベクトルとみなす）
    vecs = last_hidden_states[:, 0, :]

    # 文ベクトルから文章ベクトルを計算
    vec = aggregate_tensor(vecs)

    # 保存容量削減のため型変換
    # vec = vec.type(torch.float16)

    # データをgpuからcpuに載せ替え
    # vec = vec.cpu()

    # numpy配列に変換
    # vec = vec.numpy()

    return vec


def cos_similarity(x, y, eps=1e-8):
    """
    コサイン類似度を計算する
    """
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)


if __name__ == "__main__":
    text_a = """
夏目 漱石（なつめ そうせき、1867年2月9日〈慶応3年1月5日〉 - 1916年〈大正5年〉12月9日）は、日本の小説家、評論家、英文学者、俳人。本名は夏目 金之助（なつめ きんのすけ）。俳号は愚陀仏。明治末期から大正初期にかけて活躍した近代日本文学の頂点に立つ作家の一人である。代表作は『吾輩は猫である』『坊っちゃん』『三四郎』『それから』『こゝろ』『明暗』など。明治の文豪として日本の千円紙幣の肖像にもなり、講演録「私の個人主義」も知られている。漱石の私邸に門下生が集った会は木曜会と呼ばれた。
江戸の牛込馬場下横町（現在の東京都新宿区喜久井町）出身。大学時代に正岡子規と出会い、俳句を学ぶ。帝国大学（のちの東京帝国大学、現在の東京大学）英文科卒業後、松山で愛媛県尋常中学校教師、熊本で第五高等学校教授などを務めたあと、イギリスへ留学。帰国後は東京帝国大学講師として英文学を講じ、講義録には『文学論』がある。
講師の傍ら『吾輩は猫である』を雑誌『ホトトギス』に発表。これが評判になり『坊っちゃん』『倫敦塔』などを書く。その後朝日新聞社に入社し、『虞美人草』『三四郎』『それから』などを掲載。当初は余裕派と呼ばれた。「修善寺の大患」後は、『行人』『こゝろ』『道草』などを執筆。「則天去私（そくてんきょし）」の境地に達したといわれる。晩年は胃潰瘍に悩まされ、『明暗』が絶筆となった。
"""

    text_b = """
森 鷗外（もり おうがい、文久2年1月19日[1]〈1862年2月17日[2][注釈 1]〉 - 1922年〈大正11年〉7月9日）は、日本の明治・大正期の小説家、評論家、翻訳家、陸軍軍医（軍医総監＝中将相当）、官僚（高等官一等）。位階勲等は従二位・勲一等・功三級、医学博士、文学博士。本名は森 林太郎（もり りんたろう）。
石見国津和野（現：島根県津和野町）出身。東京大学医学部[注釈 2]卒業。
大学卒業後、陸軍軍医になり、陸軍省派遣留学生としてドイツでも軍医として4年過ごした。帰国後、訳詩編「於母影」、小説「舞姫」、翻訳「即興詩人」を発表する一方、同人たちと文芸雑誌『しがらみ草紙』を創刊して文筆活動に入った。その後、日清戦争出征や小倉転勤などにより一時期創作活動から遠ざかったものの、『スバル』創刊後に「ヰタ・セクスアリス」「雁」などを発表。乃木希典の殉死に影響されて「興津弥五右衛門の遺書」を発表後、「阿部一族」「高瀬舟」など歴史小説や史伝「澁江抽斎」なども執筆した。
晩年、帝室博物館（現在の東京国立博物館・奈良国立博物館・京都国立博物館等）総長や帝国美術院（現：日本芸術院）初代院長なども歴任した。    
"""

    vec_a = embedding(text_a)
    vec_b = embedding(text_b)
    sim = cos_similarity(vec_a, vec_b).item()

    p(vec_a.shape)
    p(vec_a.dtype)
    p(vec_a[:5])

    p(vec_b.shape)
    p(vec_b.dtype)
    p(vec_b[:5])

    p(sim)
```

## サンプルコードの実行

```sh
$ python main.py

False
device(type='cpu')
torch.Size([1536])
torch.float32
tensor([-0.1967,  0.9231,  0.3203,  0.4452,  0.1927])
torch.Size([1536])
torch.float32
tensor([-0.1913,  0.5956,  0.1263,  0.3581, -0.2445])
0.9690771102905273
```