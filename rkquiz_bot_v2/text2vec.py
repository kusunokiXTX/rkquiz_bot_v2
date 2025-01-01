import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # OpenAIクラスを使用して初期化

def get_embedding(text, model="text-embedding-3-small"):
    """指定されたテキストの埋め込みを取得する関数"""
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embeddingの取得中にエラーが発生しました: {e}")
        return None

def update_embeddings_in_json(file_path):
    """指定されたJSONファイル内のテキスト項目に対して埋め込みを更新する関数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各項目のテキストを埋め込みに変換
    for key in data:
        if isinstance(data[key], dict) and 'text' in data[key]:
            text = data[key]['text']
            data[key]['embedding'] = get_embedding(text)
        elif isinstance(data[key], list):
            for item in data[key]:
                if 'text' in item:
                    text = item['text']
                    item['embedding'] = get_embedding(text)

    # 更新されたデータをファイルに書き込む
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用例
update_embeddings_in_json('horizontal-bot/rkquiz_bot_v2/questions/0.json')
