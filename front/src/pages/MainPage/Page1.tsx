import LazyLoad from "react-lazy-load";
import logo from "../../assets/icons/logo.svg";
import main2 from "../../assets/images/main.webp";
import main from "../../assets/images/mainImg2.png";
import { Link } from "react-router-dom";
import logo2 from "../../assets/icons/SignLanguage2.png";

const Page1 = () => {
  return (
    <div
      id="section1"
      className="h-[100vh] flex flex-col md:flex-row items-center justify-center w-full"
    >
      <div className="flex flex-col items-start justify-start mt-16">
        <p className="font-bold text-black text-[2rem] mb-[10px] font-main">
          내 손 안의 작은 수어 통역가
        </p>
        <LazyLoad>
          <img src={logo2} alt="logo2" className="w-[190px] mb-[20px]" />
        </LazyLoad>

        <p className="text-lg text-gray-400 font-main mb-[10px]">
          Signlanguage와 함께라면
          <br /> 언제든 수어 번역 서비스를 이용하실 수 있습니다.
          <br/> 의료진과 농인 환자 사이의 원활한 소통을 돕기 위해 제공되고 있습니다. 
        </p>
        <Link to="/translate">
        <button className="bg-blue-500 text-white font-main text-base rounded-md mb-[10px] w-[200px] h-[50px] ">
      번역 서비스 사용해보기
      </button>

        </Link>
      </div>
      <img
        src={main}
        alt="main"
        className="hidden md:block w-auto md:h-[17rem] lg:h-[20rem] 2xl:h-[25rem] ml-[120px] mt-[60px]"
      />
    </div>
  );
};

export default Page1;
