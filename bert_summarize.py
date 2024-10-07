from transformers import pipeline

# BERTベースのモデルを使用した要約パイプラインの作成
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 要約するための長文
text = """
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. It is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context. BERT can be used for a variety of natural language understanding tasks, including question answering, text classification, and more.
"""

# 要約の実行
summary = summarizer(text, max_length=5000, min_length=25, do_sample=False)

# 要約結果の表示
print("要約結果:")
print(summary[0]['summary_text'])
