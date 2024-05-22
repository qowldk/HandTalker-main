import main32 from "../../assets/images/main2.webp";
import logo from "../../assets/icons/logo.svg";
import logo2 from "../../assets/icons/SignLanguage2.png";
import main2 from "../../assets/images/mainImg2.png";

const Page2 = () => {
  return (
    <div
      id="section2"
      className="w-full h-[100vh] flex flex-row items-center justify-start md:space-x-20 px-10 md:pr-20"
    >
      <img
        src={main2}
        alt="main2"
        className="hidden md:block w-auto max-w-full md:h-[23rem] lg:h-[27rem] xl:h-[30rem]"
      />

      <div className="md:min-w-[26rem] md:w-[28rem] lg:w-[35rem] xl:w-[43rem] flex flex-col">
        <div className="flex flex-row items-center justify-start">
          <img
            src={logo2}
            alt="logo"
            className="block w-auto h-[2.1rem] md:h-[2.5rem]"
          />
          <p className="text-3xl md:text-4xl font-bold text-black font-main ml-[15px]">
            소개
          </p>
        </div>
        <p className="text-base text-gray-400 font-main mt-[20px] leading-8">
  SignLanguage는 인공지능 기반의 혁신적인 기술을 활용하여 수어를 신속하고 정확하게 인식하여 한국어로 번역해주는 탁월한 서비스입니다. 병원에서 주로 이용되는 수어 번역 서비스로, 의료진과 농인 환자 간의 원활한 소통을 돕습니다.
</p>
<p className="hidden md:block text-base text-gray-400 font-main mt-[15px] leading-8">
  병원에서의 사용은 의사와 환자 간 의사소통을 원활하게 하여 진료 과정을 보다 효율적으로 만듭니다.
</p>
<p className="text-base text-gray-400 font-main mt-[15px] leading-8">
  SignLanguage는 수어를 통한 소통의 장벽을 허물며, 의료 서비스의 품질을 향상시킵니다. 병원 내에서 농인 환자들의 의료 서비스 접근성을 높이고, 보다 효과적인 의료 소통을 가능하게 합니다.
</p>

      </div>
    </div>
  );
};

export default Page2;
