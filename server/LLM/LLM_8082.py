import asyncio
import websockets
from openai import OpenAI
import json

from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get('API_KEY')

client = OpenAI(api_key = key)
system_role = """
당신은 병원 환자의 입장에서 글을 작성하는 Assistant입니다.

입력받은 문자열 중간에 결합되지 않은 한글 자음모음들이 있을 경우, 결합된 텍스트로 바꿔야 합니다.
예를 들어서 'ㄱㅜㄱㅂㅏㅂ'을 입력이 있을 때, 대답으로 '국밥'으로 바꿉니다.
만약 'ㄴㄴㄴㅏㅏㅂㅏㅇ'처럼 단순 조합으로 아예 성립되는 단어를 만들 수 없을 때는 중복된 자음이나 모음을 제거해서 '나방'과 같이 바꿉니다.
다른 예시로, 'ㅏㅏㄴㄴㄴㅕㅇ' 이 입력될 경우, '안녕'으로 바꿉니다.

문자열 중간의 자음모음을 결합한 이후, 또는 결합이 필요 없는 경우, 입력받은 문자열을 공백을 기준으로 나눠 단어 목록을 얻습니다.
얻은 단어 목록의 순서와 흐름을 고려하여 자연스러운 한국어 문장을 만들어 반환합니다.
단어의 순서를 유지하여 문장을 만들 때, 문장이 너무 어색해진다면 단어의 순서는 일부 자연스럽게 조정 가능합니다.

응답할 때는 환자의 입장에서 문장을 구성합니다. 예를 들어, '나 맛있다 것 먹다 기분 좋다'를 입력받으면, '저는 맛있는 것을 먹고 기분이 좋습니다.'와 같이 응답합니다.
또 하나의 예로, '머리 아프다 방문하다'를 입력받으면, '머리가 아파서 방문했습니다'와 같이 응답합니다.
또 하나의 예로, '오늘 진료 가능'을 입력받으면, '오늘 진료 가능할까요?'와 같이 응답합니다.
만약 하나의 단어만 입력 받는다면 그 단어를 그대로 반환합니다.

병원에서 환자가 의사에게 물어볼만한 질문을 생각하면서 답변을 해주시면 됩니다.

"""

def LLM_API(string):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": string}
        ]
    )
    return completion.choices[0].message.content


async def generate_sentence(websocket, path):
    try:
        while True:
            message = await websocket.recv()
            print("get!", message, "<<")
            if message != "":
                data=LLM_API(message)
                if data[0]=="'" or data[0]=='"':
                    data=data[1:]
                if data[-1]=="'" or data[-1]=='"':
                    data=data[:-1]
                print("result", data)    

                result_dict = {'result': data}
                result_json = json.dumps(result_dict)           
                try:
                    if websocket.open:
                        await websocket.send(result_json)
                except Exception as e:
                    print(f"send error: {str(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        pass


start_server = websockets.serve(generate_sentence, "localhost", 8082)


async def main():
    await start_server


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()

