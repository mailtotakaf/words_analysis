import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# MeCabで文章を形態素解析し、名詞を抽出する関数
def extract_nouns(text):
    mecab = MeCab.Tagger("-Ochasen")
    mecab.parse("")  # この行はMeCabのバグ回避
    node = mecab.parseToNode(text)
    words = []

    while node:
        # 名詞だけを抽出
        if node.feature.startswith("名詞"):
            words.append(node.surface)
        node = node.next

    return words

# 文章のリストを受け取り、重要な文を抽出する関数
def summarize_text(texts, top_n=3):
    # 形態素解析して名詞のみを使う
    tokenized_texts = [' '.join(extract_nouns(text)) for text in texts]
    
    # TF-IDFベクトライザを使用して各文の特徴量を計算
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
    
    # 各文のスコアを計算（TF-IDFの和で評価）
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    # スコアが高い順にソートし、上位N件を抽出
    top_sentence_indices = sentence_scores.argsort()[-top_n:][::-1]
    
    # 上位N件の文を返す
    summarized_texts = [texts[i] for i in top_sentence_indices]
    
    return summarized_texts

# 使用例
if __name__ == "__main__":
    texts = [
        "Pythonは非常に人気のあるプログラミング言語です。",
        "日本語の自然言語処理ではMeCabがよく使われます。",
        "TF-IDFは文章要約や特徴抽出に用いられる手法の一つです。",
        "要約は、文章全体を短くまとめることが目的です。"
    ]
    
    summary = summarize_text(texts, top_n=2)
    print("要約された文章:")
    for sentence in summary:
        print(sentence)
