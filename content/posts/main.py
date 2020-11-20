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


def blocking(sentences):
    blocks = []
    block = ""

    for i, s in enumerate(sentences):
        if len(s) > 510:
            s = s[:510]

        if len(block) + len(s) > 510:
            blocks.append(block)
            block = s
        else:
            block = block + s

        if i == len(sentences) - 1:
            blocks.append(block)

    return blocks


def bert_tokenize(blocks):
    for i, b in enumerate(blocks):
        b = b.replace("。", " 。")
        if b[-1] == "。":
            b = b[:-1]
        blocks[i] = b

    encoded = tokenizer.batch_encode_plus(blocks, padding=True, add_special_tokens=True)

    input_ids = []

    for ids in encoded["input_ids"]:
        tokens = tokenizer.convert_ids_to_tokens(ids)

        if "[SEP]" in tokens:
            sep_idx = tokens.index("[SEP]")
            sep_id = ids[sep_idx]

        if "。" in tokens:
            period_idx = tokens.index("。")
            period_id = ids[period_idx]
            ids = [sep_id if id == period_id else id for id in ids]

        input_ids.append(ids)

    return input_ids


def aggregate(embeddings, pooling_strategy="REDUCE_MEAN_MAX"):
    """
    文ベクトルを集計して文章ベクトルを返却する
    """
    if isinstance(embeddings, torch.Tensor):
        if pooling_strategy == "REDUCE_MEAN":
            return torch.mean(embeddings, dim=0)
        elif pooling_strategy == "REDUCE_MAX":
            max, _ = torch.max(embeddings, dim=0)
            return max
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            max, _ = torch.max(embeddings, dim=0)
            mean = torch.mean(embeddings, dim=0)
            return torch.hstack((max, mean))
    elif isinstance(embeddings, np.ndarray):
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embeddings, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embeddings, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embeddings, axis=0), np.mean(embeddings, axis=0)]


def cos_similarity(x, y, eps=1e-8):
    """
    コサイン類似度を計算する
    """
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)


def embedding(text, pooling_strategy="REDUCE_MEAN_MAX", debug=False):
    # 前処理
    text = preprocess(text)

    # 文単位に分割
    sentences = split_to_sentences(text)

    blocks = blocking(sentences)

    # BERTトークン化
    # encoded = tokenizer.batch_encode_plus(
    #     sentences, padding=True, add_special_tokens=True
    # )

    # BERTトークンID列を抽出
    input_ids = torch.tensor(bert_tokenize(blocks), device=device)
    # input_ids = torch.tensor(encoded["input_ids"], device=device)

    # BERTの最大許容トークン数が512なので超える場合は切り詰める
    input_ids = input_ids[:, :512]

    if debug:
        for i, s in enumerate(blocks):
            print(s)
            print(tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))

    with torch.no_grad():  # 勾配計算なし
        # 単語ベクトルを計算
        outputs = model(input_ids)

    # 最終層の隠れ状態ベクトルを取得
    last_hidden_states = outputs[0]

    # 各文における[CLS]トークンの単語ベクトルを抽出（これを文ベクトルとみなす）
    vecs = last_hidden_states[:, 0, :]

    # 文ベクトルから文章ベクトルを計算
    vec = aggregate_tensor(vecs, pooling_strategy=pooling_strategy)

    # 保存容量削減のため型変換
    # vec = vec.type(torch.float16)

    # データをgpuからcpuに載せ替え
    # vec = vec.cpu()

    # numpy配列に変換
    # vec = vec.numpy()

    return vec


if __name__ == "__main__":
    text_a = """
夏目 漱石（なつめ そうせき、1867年2月9日〈慶応3年1月5日〉 - 1916年〈大正5年〉12月9日）は、日本の小説家、評論家、英文学者、俳人。本名は夏目 金之助（なつめ きんのすけ）。俳号は愚陀仏。明治末期から大正初期にかけて活躍した近代日本文学の頂点に立つ作家の一人である。代表作は『吾輩は猫である』『坊っちゃん』『三四郎』『それから』『こゝろ』『明暗』など。明治の文豪として日本の千円紙幣の肖像にもなり、講演録「私の個人主義」も知られている。漱石の私邸に門下生が集った会は木曜会と呼ばれた。
江戸の牛込馬場下横町（現在の東京都新宿区喜久井町）出身。大学時代に正岡子規と出会い、俳句を学ぶ。帝国大学（のちの東京帝国大学、現在の東京大学）英文科卒業後、松山で愛媛県尋常中学校教師、熊本で第五高等学校教授などを務めたあと、イギリスへ留学。帰国後は東京帝国大学講師として英文学を講じ、講義録には『文学論』がある。
講師の傍ら『吾輩は猫である』を雑誌『ホトトギス』に発表。これが評判になり『坊っちゃん』『倫敦塔』などを書く。その後朝日新聞社に入社し、『虞美人草』『三四郎』『それから』などを掲載。当初は余裕派と呼ばれた。「修善寺の大患」後は、『行人』『こゝろ』『道草』などを執筆。「則天去私（そくてんきょし）」の境地に達したといわれる。晩年は胃潰瘍に悩まされ、『明暗』が絶筆となった。
"""

    text_b = """
森 鷗外（もり おうがい、文久2年1月19日[1]〈1862年2月17日[2][注釈 1]〉 - 1922年〈大正11年〉7月9日）は、日本の明治・大正期の小説家、評論家、翻訳家、陸軍軍医（軍医総監＝中将相当）、
官僚（高等官一等）。位階勲等は従二位・勲一等・功三級、医学
博士、文学博士。本名は森 林太郎（もり りんたろう）。

石見国津和野（現：島根県津和野町）出身。東京大学医学部[注釈 2]卒業。

大学卒業後、陸軍軍医になり、陸軍省派遣留学生としてドイツでも軍医として4年過ごした。帰国後、訳詩編「於母影」、小説「舞姫」、翻訳「即興詩人」を発表する一方、同人たちと文芸雑誌『しがらみ草紙』を創
刊して文筆活動に入った。その後、日清戦争出征や小倉転勤などにより一時期創作活動から遠ざかったものの、『スバル』創刊後に「ヰタ・セクスアリス」「雁」などを発表。乃木希典の殉死に影響されて「興津弥五右衛門の遺書」を発表後
、「阿部一族」「高瀬舟」など歴史小説や史伝「澁江抽斎」なども執筆した。

晩年、帝室博物館（現在の東京国立博物館・奈良国立博物館・京都国立博物館等）総長や帝国美術院（現：日本芸術院）初代院長なども歴任した。"""

    text_c = """
の方法により請求がなされるのが通常であり，これによって実質的な勝敗が決まる（門口正人編『新・裁判実務大系第11巻会社訴訟・商事仮処分・商事非訟』（青林書院・2001）252頁〔古閑裕二〕）。
募集株式の発行等の差止めの訴えおよびその仮処分の申立ては，株主が行うことができる。株式の譲渡を受けた株主については，株主名簿に株主として記載されていること（法130条，会社法大系第４巻286頁以下〔真鍋美穂子〕）または個別株主通知がされた後４週間以内に
行使すること（社債株式振替法154条２項，下記第７章１4参照）が必要である。仮処分の申立ての管轄は，本案の管轄裁判所（民事保全法12条１項），すなわち被告である会社の本店所在地を管轄する地方裁判所にある（民事訴訟法４条１項，４項）。申立ての趣旨は，発行等が
されようとしている募集株式を特定して，それを仮に差し止める旨が表現される必要があり，例えば「債務者が平成○年○月○日に開催した取締役会の決議に基づき現に手続中の普通株式○○株の募集株式の発行を仮に差し止める。」のように記載される。発行等の差止めの仮処分は仮
の地位を定める仮処分であるため，口頭弁論または債務者が立ち会うことができる審尋の期日を経る必要があり（民事保全法23条４項），株主に生ずる著しい損害または急迫の危険を避けるためこれを必要とするときに仮処分命令が発せられる（民事保全法23条２項）（以上につき
前掲『新・裁判実務大系第11巻』253頁以下〔古閑裕二〕）。公開会社が取締役会の決議に基づいて募集株式の発行等を行う場合は，当該発行等について通知・公告等（法201条３項から５項まで）が行われた時に初めて株主に発行等の事実が分かることが多い。他方，通知・公告等は
払込期日の２週間前までに行われれば良いので，仮処分の申立ておよび仮処分命令に関する決定はその間に行われることとなり，極めて短期間で手続を進行させることになる（前掲『新・裁判実務大系第11巻』254頁〔古閑裕二〕）。保全命令を出す際に，債権者（株主）に担保を立てさ
せることができ（民事保全法14条１項），実務上，保証金を立てさせることが通常である。その金額は，理論的には発行等の差止めによって会社が受けると考えられる損害の額（資金調達コスト）
"""

    vec_a = embedding(text_a, pooling_strategy="REDUCE_MEAN_MAX", debug=True)
    vec_b = embedding(text_b, pooling_strategy="REDUCE_MEAN_MAX", debug=True)
    vec_c = embedding(text_c, pooling_strategy="REDUCE_MEAN_MAX", debug=True)
    sim_a_b = cos_similarity(vec_a, vec_b).item()
    sim_b_c = cos_similarity(vec_b, vec_c).item()
    sim_c_a = cos_similarity(vec_c, vec_a).item()

    p(sim_a_b)
    p(sim_b_c)
    p(sim_c_a)