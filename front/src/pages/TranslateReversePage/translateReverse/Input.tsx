// input 의미 없은 파일임. 혹시 몰라 일단 냅둠

import React, { ChangeEvent, useState, useEffect, useRef } from "react";

const Input = () => {
  const [inputText, setInputText] = useState(""); // 사용자가 입력한 텍스트 상태
  const [ws, setWs] = useState<WebSocket | null>(null); // WebSocket 상태

  // WebSocket 서버에 연결하는 함수
  const connectWebSocket = () => {
    const socket = new WebSocket("ws://localhost:8082"); // WebSocket 서버 주소
    socket.onopen = () => {
      console.log("WebSocket connected");
      setWs(socket); // WebSocket 상태 업데이트
    };
    socket.onclose = () => {
      console.log("WebSocket disconnected");
      setWs(null); // WebSocket 상태 초기화
    };
    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
    socket.onmessage = (event) => {
      console.log("Message received from server:", event.data);
      // 서버로부터 받은 메시지를 처리하는 코드를 여기에 추가
    };
  };

  // 컴포넌트가 마운트될 때 WebSocket 연결
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws !== null) {
        ws.close(); // 컴포넌트가 언마운트될 때 WebSocket 연결 종료
      }
    };
  }, []);

  // 입력된 텍스트가 변경될 때마다 호출되는 함수
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(event.target.value); // 입력된 텍스트 상태 업데이트
  };

  // 텍스트를 WebSocket을 통해 서버로 전송하는 함수
  const sendTextViaWebSocket = () => {
    if (ws !== null && inputText.trim() !== "") {
      // WebSocket이 연결되어 있고 입력된 텍스트가 비어있지 않은 경우
      ws.send(inputText); // WebSocket을 통해 텍스트 전송
      setInputText(""); // 입력된 텍스트 초기화
    }
  };

  return (
    <div>
      <input
        type="text"
        value={inputText}
        onChange={handleInputChange}
        placeholder="입력하세요..."
      />
      <button onClick={sendTextViaWebSocket}>전송</button>
    </div>
  );
};

export default Input;
