import React, { useState, useEffect } from "react";
import axios from "axios";
import { FaArrowRightLong } from "react-icons/fa6";
import discord from "../../../assets/icons/discord.webp";
import { BiCopy, BiRevision } from "react-icons/bi";
import Swal from "sweetalert2";
import ConfigModal from "../config/ConfigModal";
import {
  translateReverseState,
  resultText,
  dchannel,
} from "../../../utils/recoil/atom";
import { useRecoilState, useRecoilValue } from "recoil";

const TranslateReverse = () => {
  const [openModal, setOpenModal] = useState(false);
  const onModalAlert = () => {
    setOpenModal(!openModal);
  };

  const [translateReverse, setTranslateReverse] = useRecoilState(
    translateReverseState
  );
  const onClick = () => {
    setTranslateReverse(true);
  };

  const [text, setText] = useRecoilState(resultText);

  const channel = useRecoilValue(dchannel);

  const SendMessage = () => {
    if (!(channel === "")) {
      axios
        .post("https://localhost:3001/api/send_message", {
          message: text,
          CHANNEL_ID: channel,
        })
        .then((response) => console.log(response.data))
        .catch((e) => console.error(e));
    }
  };

  const copyToClipboardHandler = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      Toast.fire({
        icon: "success",
        title: "클립보드에 복사되었습니다!",
      });
    } catch (e) {
      console.error(e);
    }
  };

  const textClearHandler = () => {
    setText("");
  };

  const Toast = Swal.mixin({
    toast: true,
    position: "top-right",
    showConfirmButton: false,
    timer: 1500,
    timerProgressBar: true,
    didOpen: (toast) => {
      toast.addEventListener("mouseenter", Swal.stopTimer);
      toast.addEventListener("mouseleave", Swal.resumeTimer);
    },
  });

  const [inputText, setInputText] = useState(""); // 사용자가 입력한 텍스트 상태
  const [videoUrl, setVideoUrl] = useState(""); // 비디오 URL 상태

  // 입력된 텍스트가 변경될 때마다 호출되는 함수
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(event.target.value);
  };

  const fetchVideo = async () => {
    try {
      // 비디오 파일의 상대 경로를 설정합니다.
      const videoPath = `videos/${inputText}.mp4`;

      // 비디오 URL 상태를 해당 경로로 설정합니다.
      setVideoUrl(videoPath);
    } catch (error) {
      console.error("Error fetching video:", error);
    }
  };

  return (
    <div className="mt-[5rem] md:mt-[7rem] flex flex-col items-center justify-start w-full h-full mx-auto mb-[4rem]">
      {openModal && <ConfigModal onOpenModal={onModalAlert} />}
      <button
        onClick={onModalAlert}
        className="w-[8rem] h-[2.5rem] md:w-[160px] md:h-[3rem] font-main text-xl font-bold items-end justify-end ml-[12rem] md:ml-[48.7rem] xl:ml-[67.5rem] text-white bg-main-2 rounded-lg"
      >
        연동 설정
      </button>
      <div className="flex flex-col items-center justify-center mt-2 md:flex-row">
        <div className="w-full md:w-[20rem] h-[13rem] md:w-[25rem] xl:w-[31.25rem] md:h-[31.25rem] xl:h-[37.5rem] bg-white rounded-xl border border-gray-200 shadow-md flex flex-col items-center justify-center">
          <div
            style={{
              marginBottom: "1rem",
              marginTop: "1rem", // 위에 여백 추가
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <p
              style={{
                fontSize: "0.8rem",
                color: "#999",
                marginBottom: "0.5rem",
              }}
            >
              public/videos에 있는 단어만 가능
            </p>
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <input
                type="text"
                value={inputText}
                onChange={handleInputChange}
                placeholder="단어를 입력하세요"
                style={{
                  flex: "1", // input이 여백을 채우도록 스타일 추가
                  padding: "0.5rem",
                  fontSize: "1rem",
                  border: "1px solid #ccc",
                  borderRadius: "5px",
                }}
              />
              <button
                onClick={fetchVideo}
                style={{
                  padding: "0.5rem 1rem",
                  fontSize: "1rem",
                  backgroundColor: "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer",
                }}
              >
                비디오 가져오기
              </button>
            </div>
          </div>
        </div>

        <p className="hidden md:block ml-[40px] text-6xl text-gray-200">
          <FaArrowRightLong />
        </p>

        <div
          className="mt-3 md:mt-0 flex flex-col md:ml-[40px] w-[20rem] h-[13rem] md:w-[25rem] xl:w-[31.25rem] md:h-[31.25rem] xl:h-[37.5rem] bg-white rounded-xl border border-gray-200 shadow-md"
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div
            className="video-container"
            style={{
              flex: "1",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {videoUrl && (
              <video
                controls
                autoPlay
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "auto",
                  height: "auto",
                }}
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
          </div>
          <div
            className={`flex flex-row items-center justify-end md:justify-center mr-2 md:mr-0 h-[50px] md:mb-[10px] xl:mt-[35px]`}
          >
            <button
              onClick={() => {
                copyToClipboardHandler(text);
              }}
              className="text-2xl md:text-4xl text-gray-300 mr-[15px] hover:bg-gray-200 hover:bg-opacity-30 rounded-full cursor-pointer"
            >
              <BiCopy />
            </button>
            <button
              onClick={textClearHandler}
              className="text-2xl md:text-4xl text-gray-300 md:mr-[30px] xl:mr-[75px] hover:bg-gray-200 hover:bg-opacity-30 rounded-full cursor-pointer"
            >
              <BiRevision />
            </button>
            <button
              onClick={SendMessage}
              className="ml-4 md:ml-0 flex flex-row justify-center items-center rounded-xl md:min-w-[16rem] w-[11rem] h-[2.7rem] md:w-[16rem] xl:w-[18.75rem] md:h-[3rem] xl:h-[3.2rem] bg-[#5865f2] text-white font-main text-xl"
            >
              <img
                src={discord}
                alt="discord"
                className="w-[30px] md:w-[40px] xl:w-[50px] mr-[5px]"
              />
              디스코드로 전송
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslateReverse;
