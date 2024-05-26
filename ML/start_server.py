import asyncio
import threading
import websockets
from signlang.main import run_server as start_signlang_server
from fingerspell.main_8081 import finger_spell
from LLM.LLM_8082 import make_sentence

async def main():
    # 프레임 처리기 스레드 시작 및 웹소켓 서버 실행
    processor_thread = threading.Thread(target=start_signlang_server)
    processor_thread.start()

    # 비동기 서버 시작
    await asyncio.gather(
        finger_spell_server(),
        make_sentence_server()
    )

async def finger_spell_server():
    # 웹소켓 서버 설정
    async with websockets.serve(finger_spell, "localhost", 8081):
        await asyncio.Future()  # 서버가 계속 실행되도록 유지

async def make_sentence_server():
    # 웹소켓 서버 설정
    async with websockets.serve(make_sentence, "localhost", 8082):
        await asyncio.Future()  # 서버가 계속 실행되도록 유지

if __name__ == "__main__":
    asyncio.run(main())
