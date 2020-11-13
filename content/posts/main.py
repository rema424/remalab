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


def convert_to_single_embedding(embeddings, pooling_strategy="REDUCE_MEAN"):
    if pooling_strategy == "REDUCE_MEAN":
        return torch.mean(embeddings, dim=0)
    elif pooling_strategy == "REDUCE_MAX":
        return torch.max(embeddings, dim=0)


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
    vec = convert_to_single_embedding(vecs)

    # 保存容量削減のため型変換
    vec = vec.type(torch.float16)

    # データをgpuからcpuに載せ替え
    vec = vec.cpu()

    # numpy配列に変換
    vec = vec.numpy()

    return vec


if __name__ == "__main__":
    text = """
夏目 漱石（なつめ そうせき、1867年2月9日〈慶応3年1月5日〉 - 1916年〈大正5年〉12月9日）は、日本の小説家、評論家、英文学者、俳人。本名は夏目 金之助（なつめ きんのすけ）。俳号は愚陀仏。明治末期から大正初期にかけて活躍した近代日本文学の頂点に立つ作家の一人である。代表作は『吾輩は猫である』『坊っちゃん』『三四郎』『それから』『こゝろ』『明暗』など。明治の文豪として日本の千円紙幣の肖像にもなり、講演録「私の個人主義」も知られている。漱石の私邸に門下生が集った会は木曜会と呼ばれた。
江戸の牛込馬場下横町（現在の東京都新宿区喜久井町）出身。大学時代に正岡子規と出会い、俳句を学ぶ。帝国大学（のちの東京帝国大学、現在の東京大学）英文科卒業後、松山で愛媛県尋常中学校教師、熊本で第五高等学校教授などを務めたあと、イギリスへ留学。帰国後は東京帝国大学講師として英文学を講じ、講義録には『文学論』がある。
講師の傍ら『吾輩は猫である』を雑誌『ホトトギス』に発表。これが評判になり『坊っちゃん』『倫敦塔』などを書く。その後朝日新聞社に入社し、『虞美人草』『三四郎』『それから』などを掲載。当初は余裕派と呼ばれた。「修善寺の大患」後は、『行人』『こゝろ』『道草』などを執筆。「則天去私（そくてんきょし）」の境地に達したといわれる。晩年は胃潰瘍に悩まされ、『明暗』が絶筆となった。
"""

    vec = embedding(text)
    p(vec.shape)
    p(vec.dtype)
    p(vec[:10])
