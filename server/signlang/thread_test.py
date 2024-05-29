import queue
import time
import numpy as np
import websockets
import asyncio
import threading

frame_queue = queue.Queue()


def frame_processor():
    global frame_queue

    print("START")

    while True:
        message = frame_queue.get()
        time.sleep(0.1)
        if message=="stop":
            print("큐 초기화 시작")
            time.sleep(2)
            seq = []
            frame_queue = queue.Queue() # 작업 리스트 초기화
            print("큐 초기화!")
        
async def handle_client(websocket, path):
    try:
        global frame_queue
        count=0
        while True:
            message = await websocket.recv()
            count+=1
            print("put", count, frame_queue.qsize())
            frame_queue.put(count)
            if count%50==0:
                frame_queue.put("stop")

    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        frame_queue.put(None)


start_server_1 = websockets.serve(handle_client, "localhost", 8080)

async def main():
    await asyncio.gather(start_server_1)

if __name__ == "__main__":
    processor_thread = threading.Thread(target=frame_processor)
    processor_thread.start()
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()
    processor_thread.join()



