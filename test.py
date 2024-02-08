import openai

# APIキーの設定
openai.api_key = "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" 
# openai.api_key = "incorrect-key" 

# import os
# import getpass
# # os.environ["OPENAI_API_KEY"] = getpass.getpass('OpenAI API Key:')
# os.environ["OPENAI_API_KEY"] = "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" 

# test

# GPTによる応答生成
# prompt = "以下の条件の下でおいしい食べ物を教えてください。\n条件1:和食\n条件2:甘い"
prompt = "AIが今後の経済に及ぼす影響力とまたそれによって生まれる問題点を教えてください"
response = openai.ChatCompletion.create(
                    # model = "gpt-3.5-turbo-16k-0613",
                    model = "gpt-3.5-turbo",
                   
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

# 応答の表示
print(type(response))
text = response['choices'][0]['message']['content']
print(text)

# test