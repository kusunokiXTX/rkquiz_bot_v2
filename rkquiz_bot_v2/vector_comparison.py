import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # OpenAIクラスを使用して初期化

def get_embedding(text, model="text-embedding-3-small"):
    """テキストの埋め込みを取得する関数"""
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embeddingの取得中にエラーが発生しました: {e}")
        return None
    
def cosine_similarity(vec1, vec2):
    """2つのベクトル間のコサイン類似度を計算する関数"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def load_json_embeddings(file_path):
    """JSONファイルから埋め込みデータを読み込む関数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def find_most_similar_embedding(input_vector, json_data):
    """入力ベクトルに最も類似した埋め込みを見つける関数"""
    similarities = []

    # タイトル、問題、答え、状況の埋め込みを比較
    for key in ['答え', '状況']:
        if key in json_data:  # json_dataにkeyが存在するか確認
            if isinstance(json_data[key], list):  # keyがリストの場合
                for item in json_data[key]:
                    if 'embedding' in item:  # embeddingが存在するか確認
                        embedding = item['embedding']
                        similarity = cosine_similarity(input_vector, embedding)
                        similarities.append((item['text'], similarity))
            elif 'embedding' in json_data[key]:  # keyが辞書の場合、embeddingが存在するか確認
                embedding = json_data[key]['embedding']
                similarity = cosine_similarity(input_vector, embedding)
                similarities.append((json_data[key]['text'], similarity))

    # 類似度でソート
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 最も類似度の高いものを返す
    return similarities[0] if similarities else (None, 0)

# 使用例
"""
json_data = load_json_embeddings('horizontal-bot/questions.json')
input_vector = get_embedding("相手はアイドルですか？")
most_similar_text, similarity_score = find_most_similar_embedding(input_vector, json_data)

print(f"最も類似したテキスト: {most_similar_text}, 類似度スコア: {similarity_score:.4f}")
"""
