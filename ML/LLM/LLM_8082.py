import asyncio
import websockets
from openai import OpenAI
import json

client = OpenAI(api_key="따로 입력")
system_role = """
입력받은 문자열 중간에 결합되지 않은 한글 자음모음들이 있을 경우, 결합된 텍스트로 바꾼다.
예를 들어서 'ㄱㅜㄱㅂㅏㅂ'을 입력이 있을 때, 대답으로 '국밥' 으로 바꾼다.
만약 'ㄴㄴㄴㅏㅏㅂㅏㅇ' 처럼 단순 조합으로 아예 성립되는 단어를 만들 수 없을 때는 중복된 자음이나 모음을 제거해서 '나방'과 같이 바꾼다.
다른 예시로, 'ㅏㅏㄴㄴㄴㅕㅇ' 이 입력될 경우, '안녕'으로 바꿀 수 있다.
문자열 중간의 자음모음을 결합한 이후, 혹은 없을 경우, 아래와 같은 절차를 따라 응답한다.

입력받은 문자열을 공백을 기준으로 나눠 단어 목록을 얻는다.
얻은 단어 목록의 순서와 흐름을 고려하여 자연스러운 한국어 문장을 만들어 반환한다.
단어의 순서를 유지하여 문장을 만들 때, 문장이 너무 어색해진다면 단어의 순서는 일부 자연스럽게 조정 가능하다.

예를 들어 '나 맛있다 것 먹다 기분 좋다' 를 입력받으면, '나는 맛있는 것을 먹어 기분이 좋다' 를 응답하면 된다.
또 하나의 예로, '만나다 반갑다 내일 무엇 하다' 를 입력받으면, '만나서 반갑다. 내일은 무엇을 하나?' 를 응답한다.
또 하나의 예로, '어제 놓다 오다 것 교과서'를 입력 받으면, '어제 놓고 온 것은 교과서다.'를 응답한다.
만약 하나의 단어만 입력 받는다면 그 단어를 그대로 반환한다.
최종 응답으로는 어떠한 부연 설명 없이 큰 따옴표(")나 작은 따옴표(')로 절대 감싸지 않고 반환 결과 텍스트 그대로만 응답한다. 
"""

def LLM_API(string):
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_role},
            # {"role": "user", "content": "만나다 반갑다 내일 무엇 하다"}
            # {"role": "user", "content": "안녕하세요 만나다 반갑다"}
            # {"role": "user", "content": "오늘 식당 가다 돈가스 매우 맛있다"}
            {"role": "user", "content": string}
        ]
    )
    return completion.choices[0].message.content


async def make_sentence(websocket, path):
    try:
        while True:
            message = await websocket.recv()
            print("get!", message, "<<")
            if message != "":
                # gpt 처리 로직
                print("debug1", message)                
                data=LLM_API(message)
                print("debug2", data)     

                result_dict = {'result': data}
                result_json = json.dumps(result_dict)           
                try:
                    if websocket.open:
                        await websocket.send(result_json)
                except Exception as e:
                    print(f"send error: {str(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        pass


start_server = websockets.serve(make_sentence, "localhost", 8082)


async def main():
    await start_server


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()

