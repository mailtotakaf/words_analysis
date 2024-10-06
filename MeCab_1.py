import pandas as pd
import MeCab
from sklearn.feature_extraction.text import CountVectorizer

# サンプルのデータ
data = {
    'text': [
        'この商品はとても良いです。',
        '全くおすすめできない商品です。',
        '素晴らしいパフォーマンスです。',
        '最悪の経験でした。',
        'もう一度購入したいです。',
        'お金の無駄です。'
    ]
}

# データフレームに変換
df = pd.DataFrame(data)

# MeCabを使って日本語テキストをトークン化する関数
def tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip()

# テキストをトークン化
df['tokenized_text'] = df['text'].apply(tokenize)

# CountVectorizerを使ってキーワードの頻度を計算
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['tokenized_text'])

# 単語リストを取得
words = vectorizer.get_feature_names_out()

# 単語の頻度を取得
word_count = X.toarray().sum(axis=0)

# 単語とその頻度をDataFrameにまとめる
word_freq_df = pd.DataFrame({'word': words, 'frequency': word_count})

# 頻度の高い順にソート
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

# 結果を表示
print(word_freq_df)
