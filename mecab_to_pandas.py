import MeCab
import pandas as pd

# MeCabを使って文章を形態素解析し、結果をDataFrameに格納する関数
def mecab_to_dataframe(text):
    # MeCabのTaggerを初期化
    mecab = MeCab.Tagger("-Ochasen")
    
    # MeCabのバグ回避のためのコード
    mecab.parse("")
    
    # 形態素解析結果をパース
    node = mecab.parseToNode(text)
    
    # 解析結果を格納するリスト
    words = []
    
    # 形態素解析結果の各ノードを走査
    while node:
        # ノードの表層形と特徴を取得
        word_surface = node.surface
        features = node.feature.split(',')
        
        # データが正しいもののみを対象に（BOS/EOSノードはスキップ）
        if features[0] != "BOS/EOS":
            # 必要なデータを抽出して辞書に格納
            word_info = {
                "表層形": word_surface,   # 単語
                "品詞": features[0],      # 品詞
                "品詞細分類1": features[1], # 品詞細分類1
                "品詞細分類2": features[2], # 品詞細分類2
                "品詞細分類3": features[3], # 品詞細分類3
                "原形": features[6],      # 原形
                "読み": features[7] if len(features) > 7 else None,  # 読み
                "発音": features[8] if len(features) > 8 else None   # 発音
            }
            words.append(word_info)
        
        # 次のノードへ
        node = node.next
    
    # PandasのDataFrameに変換
    df = pd.DataFrame(words)
    return df

# 解析する文章
text = "Pythonは非常に人気のあるプログラミング言語です。"

# MeCabで解析し、DataFrameに変換
df = mecab_to_dataframe(text)

# DataFrameの表示
print(df)
