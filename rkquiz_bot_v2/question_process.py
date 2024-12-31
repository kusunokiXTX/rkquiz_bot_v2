import json
import os
import random
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
        situation_list = []
        for i, answer in enumerate(json_data["状況"], 1):
            answer_text = answer["text"] if isinstance(answer, dict) else answer
            situation_list.append(f"{i}. {answer_text}")
                
        situation_prompt = ""
        for i, situation_text in enumerate(situation_list, 1):
            situation_prompt += f"{i}. {situation_text}\n"

        prompt = (
            f"#命令\n"
            f"以下の質問と状況の内容を比較し、質問が状況の内容と一致するかを判断してください。\n"
            f"判断基準：\n"
            f"- はい：質問の内容が状況と矛盾なく合致している場合。\n"
            f"- いいえ：質問の内容が状況と矛盾している、または状況から判断できない場合。\n"
            f"\n"
            f"#問題\n"
            f"{json_data['問題'][0]['text']}\n"
            f"\n"
            f"#状況\n"
            f"{situation_prompt}"
            f"\n"
            f"#質問\n"
            f"{input_text}\n"
            f"\n"
            f"#詳細な判断基準\n"
            f"**質問の肯定形・否定形の判断:**\n"
            f"   - **肯定形:** 質問文中に否定辞（例：「ない」「ません」「～ではない」など）を含まない文。\n"
            f"   - **否定形:** 質問文中に否定辞（例：「ない」「ません」「～ではない」など）を含む文。\n"
            f"   ※ただし、反語的な表現には注意してください。\n"
            f"\n"
            f"**否定形の扱い:**\n"
            f"   - **手順:** まず、質問が肯定形であるか否定形であるかを判断してください。\n"
            f"   - 質問が否定形の場合：状況が肯定的なら「いいえ」、否定的なら「はい」と判断します。\n"
            f"   - 質問が肯定形の場合：状況が肯定的なら「はい」、否定的なら「いいえ」と判断します。\n"
            f"   - **反語的な表現への注意:** 文脈を考慮し、字面通りの肯定否定で判断しないこと。\n"
            f"\n"
            f"**状況と質問内容の一致に関する詳細な判断指針:**\n"
            f"以下の質問例を参考に、状況と質問の内容が一致するかどうかを判断してください。\n"
            f"これらの質問は、状況を様々な側面から捉えるためのものです。\n"
            f"\n"
            f"   - 質問例：\n"
            f"      - この出来事は屋内で起きましたか？\n"
            f"      - 時間帯は昼間ですか？\n"
            f"      - この人物は成人ですか？\n"
            f"      - この行動は故意的なものですか？\n"
            f"      - 他の人は現場にいましたか？\n"
            f"      - その物は日用品ですか？\n"
            f"      - 事件/出来事は1時間以内に起きましたか？\n"
            f"      - けが人は出ましたか？\n"
            f"      - お金は関係していますか？\n"
            f"      - 天候は関係していますか？\n"
            f"      - その物は電化製品ですか？\n"
            f"      - その場所は住宅地ですか？\n"
            f"      - 被害は発生しましたか？\n"
            f"      - その状況は普通の日本で起こりうることですか？\n"
            f"      - 犯罪性はありますか？\n"
            f"      - 動物は関係していますか？\n"
            f"      - 法律や規則に違反していますか？\n"
            f"      - 誰かが何かを見落としたことが原因ですか？\n"
            f"      - 特殊な職業が関係していますか？\n"
            f"      - その物は1万円以上の価値がありますか？\n"
            f"      - 感情的な要素は関係していますか？\n"
            f"      - 誤解や勘違いが発端となっていますか？\n"
            f"      - 事前に計画されたことですか？\n"
            f"      - 季節は関係していますか？\n"
            f"\n"
            f"   - **判断のポイント:**\n"
            f"      - 質問の内容が状況の説明と直接的に一致するか。\n"
            f"      - 質問の内容が状況から合理的に推測できるか。\n"
            f"      - 質問の内容が状況と矛盾しないか。\n"
            f"      - 質問が否定形の場合、状況の肯定否定と整合性が取れているか。\n"
            f"      - 上記の質問例のような具体的な観点から、状況と質問を比較検討してください。\n"
            f"      - 例：状況が「公園で犬がボールを追いかけている」で、質問が「動物は関係していますか？」の場合、「はい」と判断します。\n"
            f"      - 例：状況が「オフィスで会議をしている」で、質問が「この出来事は屋内で起きましたか？」の場合、「はい」と判断します。\n"
            f"      - 例：状況が「雨の日に人が傘をさして歩いている」で、質問が「天候は関係していますか？」の場合、「はい」と判断します。\n"
            f"      - 例：状況が「子供がおもちゃで遊んでいる」で、質問が「その物は1万円以上の価値がありますか？」の場合、状況から**明確に**判断できない場合にのみ「わからない」と判断してください。状況からどちらとも言えない場合は「いいえ」と判断してください。\n"
            f"\n"
            f"#「わからない」と判断する基準\n"
            f"以下のいずれかに該当する場合、質問に対する回答は「どちらでもない」としてください。\n"
            f"1. 状況の説明の中に、質問の内容に関する情報が明示的に記述されていない場合。\n"
            f"   - 例：「部屋の電気はついていましたか？」という質問に対し、状況設定に電気の点灯に関する記述がない場合。\n"
            f"   - 例：「犯人は男性でしたか？」という質問に対し、状況設定に犯人の性別に関する記述がない場合。\n"
            f"2. 質問の内容が、状況における出来事や謎解きの核心部分に影響を与えない、重要でない情報である場合。\n"
            f"   - 例：「その出来事は昼間に起こりましたか？」という質問に対し、時間帯が謎解きに影響しない場合。\n"
            f"   - 例：「その人は右利きですか？」という質問に対し、利き手が問題解決に関係ない場合。\n"
            f"3. 質問の内容について、状況から肯定・否定どちらとも断定できない場合。\n"
            f"   - 例：「被害額はいくらでしたか？」という質問に対し、金額に関する具体的な情報がない、または推測も困難な場合。\n"
            f"   - 例：「現場に居合わせた人は何人いましたか？」という質問に対し、人数の記述がなく、人数が状況把握に必須の情報でない場合。\n"
            f"\n"
            f"#回答形式\n"
            f"回答：[はい/いいえ/わからない]\n"
            f"\n"
        )
        return prompt
        
    except KeyError as e:
        raise ValueError(f"Invalid JSON structure: missing key {str(e)}")

def evaluate_answer(client, model, prompt: str) -> Dict[str, str]:
    """Evaluate answer using OpenAI API."""
    try:
        messages = [
            {"role": "system", "content": f"{prompt}\n\n回答は「はい」「いいえ」または「わからない」のいずれかで答えてください。"},
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
                            "enum": ["はい", "いいえ", "わからない"]
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

def question_process(json_data, input_text: str, model: str):
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
#print(question_process(json_data, "好きな人の名前は？", "gpt-4o-mini"))