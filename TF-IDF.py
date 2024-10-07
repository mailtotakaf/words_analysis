import MeCab
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# MeCabの初期化
mecab = MeCab.Tagger('-Owakati')  # 分かち書きの出力形式

# サンプル文書
documents = [
    "私は学生です。",
    "学生は勉強が大切です。",
    "プログラミングは楽しいです。",
    "数学の授業が好きです。",
    "勉強は重要です。",
    "私はプログラミングを学んでいます。",
    "学生としての生活は多様です。"
]

# 形態素解析を行う関数
def tokenize(text):
    return mecab.parse(text).strip()

# 各文書に対して形態素解析を適用
tokenized_documents = [tokenize(doc) for doc in documents]

# TF-IDFベクトライザの初期化
tfidf_vectorizer = TfidfVectorizer()

# 文書からTF-IDFを算出
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_documents)

# 単語のリストを取得
words = tfidf_vectorizer.get_feature_names_out()

# TF-IDFの行列をDataFrameに変換
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=words)

# DataFrameの表示
print(df_tfidf)
