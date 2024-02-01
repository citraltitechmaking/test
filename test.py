import openai

# APIキーの設定
openai.api_key = "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" 

# GPTによる応答生成
prompt = "以下の条件の下でおいしい食べ物を教えてください。\n条件1:和食\n条件2:甘い"
response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo-16k-0613",
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

# 応答の表示
text = response['choices'][0]['message']['content']
print(text)

# test