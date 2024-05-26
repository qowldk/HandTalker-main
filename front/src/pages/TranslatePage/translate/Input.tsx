import { useImperativeHandle, forwardRef, useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { Camera } from "@mediapipe/camera_utils";
import {
  HAND_CONNECTIONS,
  Holistic,
  Results as HolisticResults,
  POSE_CONNECTIONS,
} from "@mediapipe/holistic";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { useRecoilState } from "recoil";
import { resultText, isGeneratingSentence } from "../../../utils/recoil/atom";

export interface ChildProps {
  send_words: () => void;
}


const Input = forwardRef<ChildProps>((props, ref) => {
  const [isGenerating, setIsGenerating] = useRecoilState(isGeneratingSentence); // LLM API 응답 대기중 여부

  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasPoseRef = useRef<HTMLCanvasElement>(null);
  const resultsRef = useRef<HolisticResults | null>(null);

  const transmission_frequency = 1000/30;  // 8080 전송 주기

  const [previous, setPrevious] = useState('') // 웹소켓으로부터 받은 이전 단어

  const [loading, setLoading] = useState<boolean>(true);
  const handleUserMedia = () => setTimeout(() => setLoading(false), 1_000);
  const [text, setText] = useRecoilState(resultText);

  const [isChecked, setIsChecked] = useState(false);

  const handleCheckboxChange = () => {
    setIsChecked(!isChecked);
    setText("");
    setPrevious("");
  };

  // useEffect(()=>{console.log("디버그!!", previous)}, [previous]);
	useImperativeHandle(ref, () => ({
	  // 부모에서 사용하고자 하는 함수이름
    send_words,
	}));

  const OutputData = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    const imgData: string | undefined = imageSrc?.toString()?.substr(23);
    if (imgData && isChecked) {
      if (socketRef_hands.current.readyState === WebSocket.OPEN) {
        socketRef_hands.current.send(imgData);
      } else {
        console.error("ws connection is not open. (8081)");
        // socketRef = useRef<WebSocket>(new WebSocket("ws://localhost:8080"));
        // window.location.reload();
        setText("Server2 disconected");
        // alert("8080 closed.\n8080 웹소켓 연결이 끊겨버리는 현상이 있으며, 새로고침 시 재연결 가능하다.\n원인은 혼잡 상황 발생 또는 일정 시간 미사용 등으로 추정.");
      }
      // console.log(imgData);
    }
  }, [webcamRef, isChecked]);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    const imgData: string | undefined = imageSrc?.toString()?.substr(23);
    if (imgData && !isChecked) {
      if (socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(imgData);
      } else {
        console.error("ws connection is not open. (8080)");
        // socketRef = useRef<WebSocket>(new WebSocket("ws://localhost:8080"));
        // window.location.reload();
        setText("Server Disconnected");
        // alert("8080 closed.\n8080 웹소켓 연결이 끊겨버리는 현상이 있으며, 새로고침 시 재연결 가능하다.\n원인은 혼잡 상황 발생 또는 일정 시간 미사용 등으로 추정.");
      }
      // console.log(imgData);
    }
  }, [webcamRef, isChecked]);
  const send_words = useCallback(() => {
    // if (text && !isChecked) {
      // console.log("APICALLDEBUG1", socketRef_LLM.current.readyState, WebSocket.OPEN, socketRef_LLM.current.readyState === WebSocket.OPEN);
      if (text==='') return;
      if (socketRef_LLM.current.readyState === WebSocket.OPEN) {
        // console.log("debug_LLM", text);
        socketRef_LLM.current.send(text);
      } else {
        console.error("ws connection is not open. (8082)");
      }
    // }
  }, [text]);


  useEffect(() => {
    if (!isChecked) {
      const interval = setInterval(capture, transmission_frequency);
      return () => clearInterval(interval);
    } else {
      const interval = setInterval(OutputData, 1000);
      return () => clearInterval(interval);
    }
  }, [capture, webcamRef, OutputData, isChecked]);

  /**
   * 검출결과（프레임마다 호출됨）
   * @param results
   */
  const onPoseResults = useCallback((results: HolisticResults) => {
    const canvasElement = canvasPoseRef.current!;
    const canvasCtx = canvasElement.getContext("2d"); // 에러??
    if (canvasCtx === null) {
      return;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // Only overwrite existing pixels.
    canvasCtx.globalCompositeOperation = "source-in";
    canvasCtx.fillStyle = "#00FF00";
    canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

    // Only overwrite missing pixels.
    canvasCtx.globalCompositeOperation = "destination-atop";
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    canvasCtx.globalCompositeOperation = "source-over";
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 4,
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
      color: "#FF0000",
      lineWidth: 2,
    });
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {
      color: "#CC0000",
      lineWidth: 5,
    });
    drawLandmarks(canvasCtx, results.leftHandLandmarks, {
      color: "#00FF00",
      lineWidth: 2,
    });
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {
      color: "#00CC00",
      lineWidth: 5,
    });
    drawLandmarks(canvasCtx, results.rightHandLandmarks, {
      color: "#FF0000",
      lineWidth: 2,
    });
    canvasCtx.restore();
    resultsRef.current = results;
  }, []);

  // 초기설정
  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
      },
    });
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: true,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    holistic.onResults(onPoseResults);

    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null
    ) {
      const camera = new Camera(webcamRef.current.video!, {
        onFrame: async () => {
          // await hands.send({ image: webcamRef.current!.video! });
          await holistic.send({ image: webcamRef.current!.video! });
        },
        width: 1280,
        height: 720,
      });
      camera.start();
    }
  }, [onPoseResults]);

  const socketRef = useRef<WebSocket>(new WebSocket("ws://localhost:8080"));
  const socketRef_hands = useRef<WebSocket>(new WebSocket("ws://localhost:8081"));
  const socketRef_LLM = useRef<WebSocket>(new WebSocket("ws://localhost:8082"));
  
  socketRef_LLM.current.onmessage = (event) =>{
    console.log(`receive message(LLM): ${event.data}`);
    const jsonString = JSON.parse(event.data);
    setText(jsonString.result);
    setIsGenerating(false);
  }

  socketRef.current.onmessage = (event) => {
    console.log(`receive message: ${event.data}`);
    const jsonString = JSON.parse(event.data);
    console.log(`receive string: ${jsonString.result}`);
    if (!isChecked){
      if (previous !== jsonString.result){
        if (jsonString.result==='') return;
        setText(text + ' ' + jsonString.result);
        // console.log("before API call", text);
        // send_words();
        // console.log("after API call", text);
        setPrevious(jsonString.result)
      }      
    }
    console.log(text);
  };
  socketRef_hands.current.onmessage = (event) => {
    console.log(`receive message: ${event.data}`);
    const jsonString = JSON.parse(event.data);
    if (isChecked) {
      setText(text + jsonString.result);
    }
    console.log(text);
  };

  
  useEffect(() => {
    socketRef.current.onopen = () => {
      console.log("ws connected (8080_useEffect)");
    };
    socketRef.current.onclose = () => {
      console.log("ws closed (8080_useEffect)");
    };

    return () => {
      socketRef.current.close();
    };
  }, []);

  useEffect(() => {
    socketRef_hands.current.onopen = () => {
      console.log("ws connected (8081_useEffect)");
    };
    socketRef_hands.current.onclose = () => {
      console.log("ws closed (8081_useEffect)");
    };
    return () => {
      socketRef_hands.current.close();
    };
  }, []);

  useEffect(() => {
    socketRef_LLM.current.onopen = () => {
      console.log("ws connected (8082_useEffect)");
    };
    socketRef_LLM.current.onclose = () => {
      console.log("ws closed (8082_uesEffect)");
    };
    return () => {
      socketRef_LLM.current.close();
    };
  }, []);

  return (
    <div className="">
      {loading && (
        <div className="z-10 absolute w-[20rem] h-[20rem] md:w-[25rem] xl:w-[37.5rem] md:h-[31.25rem] xl:h-[37.5rem] rounded-xl border border-gray-200 shadow-md flex items-center justify-center">
          로딩 중...
        </div>
      )}
      <label className="absolute z-10 inline-flex items-center mt-2 ml-2 cursor-pointer select-none themeSwitcherTwo">
        <input
          type="checkbox"
          checked={isChecked}
          onChange={handleCheckboxChange}
          className="sr-only"
        />
        <span className="flex items-center text-sm font-medium text-black text-white font-main label">
          지문자 모드
        </span>
        <span
          className={`slider mx-4 flex h-8 w-[60px] items-center rounded-full p-1 duration-200 ${
            isChecked ? "bg-main-2" : "bg-[#CCCCCE]"
          }`}
        >
          <span
            className={`dot h-6 w-6 rounded-full bg-white duration-200 ${
              isChecked ? "translate-x-[28px]" : ""
            }`}
          ></span>
        </span>
      </label>
      <div className="-scale-x-100 relative w-[20rem] h-[20rem] md:w-[25rem] xl:w-[37.5rem] md:h-[31.25rem] xl:h-[37.5rem] overflow-hidden flex flex-col items-center justify-center rounded-[15px]">
        {/* 비디오 캡쳐 */}

        <Webcam
          audio={false}
          style={{
            visibility: loading ? "hidden" : "visible",
            objectFit: "fill",
            position: "absolute",
            width: "100%",
            height: "100%",
          }}
          mirrored={false}
          width={600}
          height={600}
          ref={webcamRef}
          onUserMedia={handleUserMedia}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 600, height: 600, facingMode: "user" }}
        />

        {/* 랜드마크를 손에 표시 */}
        <canvas
          ref={canvasPoseRef}
          className="absolute w-[20rem] h-[20rem] md:w-[25rem] xl:w-[37.5rem] md:h-[31.25rem] xl:h-[37.5rem] bg-white"
          width={600}
          height={600}
        />
      </div>
    </div>
  );
});

export default Input;
