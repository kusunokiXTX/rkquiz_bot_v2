import discord
from discord.ext import commands
import asyncio
import datetime
import re
import os
import openai
import random
from dotenv import load_dotenv

import answer_process
import question_process
import vector_comparison


load_dotenv()

# Discordボットのトークンと権限の設定
token = os.getenv("DISCORD_BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True  # メッセージの内容を読み取る権限を付与

# OpenAIクライアントの設定
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-2024-11-20" #gptモデルを選択する必要がある。

#jsonファイルの読み込み
json_pass = os.path.join(os.path.dirname(__file__), "questions")  # JSONファイルのパス
json_dict = answer_process.get_json_data(json_pass)  # JSONデータの取得

#feedbackの設定
feedback ={
  "high_list": ["鋭い目線ですね", "核心に迫っています", "重要な部分を見つけましたね", "良い方向に進んでいます", "その調子です", "大事なことに気づきましたね", "そこが要点です", "その推理は面白いですね", "確かな視点ですね", "筋が良い質問です"],
  "medium_list": ["もう一工夫ありそうです", "視点を少し変えてみては？", "その先にヒントがありそうです", "近づいていますよ", "考えを広げてみましょう", "そこから掘り下げてみては？", "その考えを発展させてみては？", "方向性は悪くないです", "可能性がありそうですね"],
  "low_list": ["違う側面も考えてみましょう", "その前に必要な情報がありそうです", "少し寄り道しているようです", "基本に立ち返ってみましょう", "まだ見えていない部分がありそうです", "別の切り口を探してみては？", "遠回りしているかもしれません", "方向を変えて考えてみましょう", "違う糸口を見つけてみましょう", "新しい視点で見直してみては"]
}

# ボットの設定
bot = commands.Bot(command_prefix=['!', '！'], intents=intents)  # コマンドプレフィックスの設定

class QuizState:
    def __init__(self):
        self.current_quiz_index = -1
        self.current_quiz_channel = None
        self.last_question_time = None
        self.current_question_number = 0
        self.correct_answer = None
        self.quiz_history = []

    def reset(self):
        self.current_quiz_index = -1
        self.current_quiz_channel = None
        self.last_question_time = None
        self.current_question_number = 0
        self.quiz_history = []

quiz_state = QuizState()

@bot.event
async def on_ready():
    print(f'{bot.user.name} が起動しました')

@bot.command(name='クイズ')
async def start_quiz(ctx):
    if quiz_state.current_quiz_index != -1:
        await ctx.send("すでにクイズが開始されています。")
        return
    quiz_state.current_quiz_index = 0
    quiz_state.current_quiz_channel = ctx.channel
    quiz_state.current_question_number = 1
    await send_question(ctx)

async def send_question(ctx):
    if quiz_state.current_question_number > len(json_dict):
        await ctx.send("全ての問題は終了しました。\nクイズを終了します。\nゲームをリセットしました。")
        quiz_state.reset()
        return
    
    if quiz_state.current_quiz_index < len([quiz_state.current_question_number]):
        title = json_dict[str(quiz_state.current_question_number - 1)]["タイトル"][0]["text"]
        question = json_dict[str(quiz_state.current_question_number - 1)]["問題"][0]["text"]
        await ctx.send(f"【第{quiz_state.current_question_number}問】\n\n**{title}**\n> {question}")
        quiz_state.last_question_time = datetime.datetime.now()
        quiz_state.quiz_history.append(quiz_state.current_quiz_index)
        
        bot.loop.create_task(check_timeout())

async def check_timeout():
    while quiz_state.current_quiz_index != -1 and quiz_state.current_quiz_channel is not None:
        if quiz_state.last_question_time and (datetime.datetime.now() - quiz_state.last_question_time).total_seconds() >= 300:
            await quiz_state.current_quiz_channel.send("タイムアウトによりクイズを終了します。")
            quiz_state.reset()
            return
        await asyncio.sleep(60)

@bot.command(name='スキップ')
async def skip_question(ctx):
    if quiz_state.current_quiz_index != -1:
        quiz_state.current_question_number += 1
        await send_question(ctx)
    else:
        await ctx.send("クイズが開始されていません。")

@bot.command(name='終了')
async def end_quiz(ctx):
    if quiz_state.current_quiz_index != -1:
        await ctx.send("クイズを終了します。")
        quiz_state.reset()
    else:
        await ctx.send("クイズが開始されていません。")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if re.match(r'^[=＝]', message.content):
        if quiz_state.current_quiz_index != -1 and message.channel == quiz_state.current_quiz_channel:
            print(f"答えを受け付けました: {message.content}")
            quiz_state.last_question_time = datetime.datetime.now()
            input_text_answer = message.content.strip()[1:].strip()
            input_vector_answer = vector_comparison.get_embedding(input_text_answer)
            most_similar_text_answer, similarity_score_answer = vector_comparison.find_most_similar_embedding(input_vector_answer, json_dict[str(quiz_state.current_question_number - 1)])
            answer_result = answer_process.answer_process(json_dict[str(quiz_state.current_question_number - 1)], input_text_answer, model)
            print(answer_result)
            print(similarity_score_answer)

            if answer_result['answer'] == '完全一致':
                await message.channel.send("正解です！")
                await message.channel.send(f"答え: {most_similar_text_answer}")
                quiz_state.current_question_number += 1
                await send_question(message.channel)
            elif answer_result['answer'] == '部分一致':
                if similarity_score_answer >= 0.7:
                    await message.channel.send("一部は正解です！素晴らしいですね！ただ、もう少しだけ要素が足りないようです。\n方向性の正しさ: {}% です。".format(int(similarity_score_answer * 100)))
                    await send_question(message.channel)
                elif similarity_score_answer >= 0.6:
                    await message.channel.send("一部は正解です！良い線いっていますが、もう少し深く考えてみましょう。\n方向性の正しさ: {}% です。".format(int(similarity_score_answer * 100)))
                    await send_question(message.channel)
                elif similarity_score_answer >= 0.5:
                    await message.channel.send("一部は正解です！でも、まだ足りない要素がありますね。もう少し頑張ってみてください。\n方向性の正しさ: {}% です。".format(int(similarity_score_answer * 100)))
                    await send_question(message.channel)
                elif similarity_score_answer >= 0.0:
                    await message.channel.send("一部は正解です。もう一度考えてみてください。")
                    await send_question(message.channel)
            else:
                await message.channel.send("不正解です。もう一度考えてみてください。")
                await send_question(message.channel)
        else:
            await message.channel.send("クイズが開始されていないか、クイズが行われているチャンネルではありません。")

    if re.match(r'^[?？]', message.content):
        if quiz_state.current_quiz_index != -1 and message.channel == quiz_state.current_quiz_channel:
            print(f"質問を受け付けました: {message.content}")
            quiz_state.last_question_time = datetime.datetime.now()
            input_text_question = message.content.strip()[1:].strip()
            input_vector_question = vector_comparison.get_embedding(input_text_question)
            most_similar_text_question, similarity_score_question = vector_comparison.find_most_similar_embedding(input_vector_question, json_dict[str(quiz_state.current_question_number - 1)])
            question_result = question_process.question_process(json_dict[str(quiz_state.current_question_number - 1)], input_text_question, model)
            print(question_result)
            print(similarity_score_question)

            if question_result is not None and question_result['answer'] == 'はい':
                if similarity_score_question >= 0.7:
                    feedback_index = random.randint(0, len(feedback["high_list"]) - 1)
                    await message.channel.send("はい。\n" + feedback["high_list"][feedback_index])
                elif similarity_score_question >= 0.5:
                    feedback_index = random.randint(0, len(feedback["medium_list"]) - 1)
                    await message.channel.send("はい。\n" + feedback["medium_list"][feedback_index])
                elif similarity_score_question >= 0.3:
                    feedback_index = random.randint(0, len(feedback["low_list"]) - 1)
                    await message.channel.send("はい。\n" + feedback["low_list"][feedback_index])
                elif similarity_score_question >= 0.0:
                    await message.channel.send("はい。\n" + "その質問はあまりよくないですね。")
            elif question_result is not None and question_result['answer'] == 'いいえ':
                if similarity_score_question >= 0.7:
                    feedback_index = random.randint(0, len(feedback["high_list"]) - 1)
                    await message.channel.send("いいえ。\n" + feedback["high_list"][feedback_index])
                elif similarity_score_question >= 0.5:
                    feedback_index = random.randint(0, len(feedback["medium_list"]) - 1)
                    await message.channel.send("いいえ。\n" + feedback["medium_list"][feedback_index])
                elif similarity_score_question >= 0.3:
                    feedback_index = random.randint(0, len(feedback["low_list"]) - 1)
                    await message.channel.send("いいえ。\n" + feedback["low_list"][feedback_index])
                elif similarity_score_question >= 0.0:
                    await message.channel.send("いいえ。\n" + "その質問はあまりよくないですね。")
            elif question_result is not None and question_result['answer'] == 'わからない':
                await message.channel.send("わかりません。")
            else:
                await message.channel.send("エラーが発生しました。")
        else:
            await message.channel.send("クイズが開始されていないか、クイズが行われているチャンネルではありません。")

    await bot.process_commands(message)

bot.run(token)