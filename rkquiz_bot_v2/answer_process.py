import json
import os
from typing import Optional, Dict, Any
import openai
from dotenv import load_dotenv
from pathlib import Path

def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key

def get_json_data(json_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load all JSON files from the specified directory and return them as a dictionary indexed by filename."""
    dir_path = Path(json_path)
    json_data_dict = {}
    try:
        if not dir_path.exists():
            raise FileNotFoundError(f"ディレクトリ '{json_path}' が見つかりません")
        
        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError("ディレクトリ内にJSONファイルが見つかりません")
        
        for json_file in json_files:
            with json_file.open('r', encoding='utf-8') as f:
                json_data_dict[json_file.stem] = json.load(f)  # ファイル名をキーにする
        
        """
        json_file_path = 'json_data.json'
        if not Path(json_file_path).exists():
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data_dict, json_file, ensure_ascii=False, indent=4)
        """

        return json_data_dict
            
    except Exception as e:
        print(f"JSONデータの読み込みエラー: {str(e)}")
        return None

def generate_prompt(json_data: Dict[str, Any], input_text: str) -> str:
    """Generate evaluation prompt from quiz data and input."""
    try:
        prompt = (
            "以下の手順に従って、「入力」が指定された要素を全て網羅しているかどうかを判定してください。\n"
            "1.答えの確認：\n"
            "  - 答えが水平思考クイズの解答として成立しているか確認する\n"
            "  - 入力された状況を説明できる論理的な答えになっているか確認する\n"
            "2.要素の確認：\n"
            "  - 答えから「何が起きたか」「なぜそうなったか」というストーリーの大枠を理解する上で必要な要素を抽出する\n"
            "3.入力を分析：\n"
            "  - 入力文に各要素が明示的または暗示的に含まれているか確認する\n"
            "4.欠落の確認：\n"
            "  - 要素リストの項目で、入力文から読み取れないものを列挙する\n"
            "5.判定基準：\n"
            "  - 完全一致：\n"
            "    - 状況の大枠が理解できている\n"
            "    - 答えの核となる要点（なぜそうなったのか）が表現されている\n"
            "    - （表現方法や細かい説明の有無は問わない）\n"
            "  - 部分一致：\n"
            "    - 状況の一部は理解している\n"
            "    - しかし重要な説明が不足している\n"
            "  - 不一致：\n"
            "    - 状況の理解が大きく異なる\n"
            "    - または全く別の解釈をしている\n"
            "出力形式：\n"
            "  - [完全一致/部分一致/不一致]\n"
            "問題:"
            + json_data["問題"][0]["text"] + "\n"
            "解答例:\n"
            "1." + json_data["答え"][0]["text"] + "\n" 
            "2." + json_data["答え"][1]["text"] + "\n" 
            "3." + json_data["答え"][2]["text"] + "\n" 
            "\n入力:"
            + input_text
        )
        return prompt
        
    except KeyError as e:
        raise ValueError(f"Invalid JSON structure: missing key {str(e)}")

def evaluate_answer(client, model, prompt: str) -> Dict[str, str]:
    """Evaluate answer using OpenAI API."""
    try:
        messages = [
            {"role": "system", "content": f"{prompt}\n\n回答は「完全一致」「部分一致」「不一致」のいずれかで答えてください。"},
            {"role": "user", "content": ""}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[{
                "name": "answer_check",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["完全一致", "部分一致", "不一致"]
                        }
                    },
                    "required": ["answer"]
                }
            }],
            function_call={"name": "answer_check"}
        )

        if response.choices[0].message.function_call:
            return json.loads(response.choices[0].message.function_call.arguments)
        else:
            return {"answer": "エラー", "error": "Function call がありませんでした"}

    except openai.APIError as e:
        return {"answer": "エラー", "error": f"OpenAI APIエラー: {str(e)}"}
    except Exception as e:
        return {"answer": "エラー", "error": f"予期せぬエラー: {str(e)}"}

def answer_process(json_data, input_text: str, model: str):
    try:
        api_key = load_api_key()
        client = openai.OpenAI(api_key=api_key)

        if json_data is None:
            return
        
        prompt = generate_prompt(json_data, input_text)
        result = evaluate_answer(client, model, prompt)
        return result
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

#使用例
#json_data = get_json_data("horizontal-bot/rkquiz_bot_v2/questions")["0"]
#print(json_data["答え"][0]["text"])
#print(answer_process(json_data, "好きな人の名前は？", "gpt-4o-mini"))

"""
このリストには辞書が一つしか含まれていないため、インデックス 1 にアクセスしようとするとエラーになります。
もし "タイトル" に複数の要素がある場合は、data['0']['タイトル'][0]['embedding'] で
2番目の要素の "embedding" の値を取得できます。
"""