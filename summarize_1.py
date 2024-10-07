import MeCab

def extract_keywords(text):
    # MeCabのインスタンスを作成
    mecab = MeCab.Tagger()
    
    # MeCabで形態素解析
    parsed = mecab.parse(text)
    
    # 名詞や動詞などの重要単語を抽出する
    keywords = []
    for line in parsed.splitlines():
        if line == "EOS":
            break
        word, details = line.split("\t")
        features = details.split(",")
        
        # 名詞や動詞をキーワードとして抽出
        if features[0] in ["名詞", "動詞"]:
            keywords.append(word)
    
    return keywords

def summarize(text):
    # 形態素解析で重要なキーワードを抽出
    keywords = extract_keywords(text)
    
    # 抽出したキーワードを使って要約を作成
    summary = " ".join(keywords[:min(5, len(keywords))])  # キーワードの最初の5つを使用
    return summary

if __name__ == "__main__":
    text = "明日は晴れると思いますが、今日は雨が降っています。天気予報によると、週末も雨が降る可能性が高いです。"
    
    summary = summarize(text)
    print("要約:", summary)
