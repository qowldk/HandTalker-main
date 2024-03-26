
import asyncio
import websockets
from signlang.main import handle_client
from fingerspell.main_8081 import finger_spell
from LLM.LLM_8082 import make_sentence


start_server_1 = websockets.serve(handle_client, "localhost", 8080)
start_server_2 = websockets.serve(finger_spell, "localhost", 8081)  
start_server_3 = websockets.serve(make_sentence, "localhost", 8082)


async def main():
    # await start_server_1
    await asyncio.gather(start_server_1, start_server_2, start_server_3)



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()