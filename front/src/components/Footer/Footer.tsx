//import logo from "../../assets/icons/handtalker.webp";
import { AiFillGithub } from "react-icons/ai";
import devocean from "../../assets/icons/footer_devocean.webp";
import { useLocation } from "react-router-dom";
import newLogo from "../../assets/icons/icon.webp";

const Footer = () => {
  const location = useLocation();
  const path = location.pathname;

  return (
    <div
      className={`bg-[#1F2D21] w-full ${
        path === "/" ? "fixed bottom-[0px]" : "relative"
      } h-32 flex flex-row items-center justify-center`}
    >
      <img src={newLogo} alt="logo" className="w-auto h-20 md:h-32" />
      <div className="flex flex-col items-start justify-start ml-[30px]">
        <p className="font-main text-white text-xs md:text-sm">
          ©2024 Computer Science and Engineering
        </p>
        <p className="font-main text-white text-xs md:text-sm">
           SENIER PROJECT 
        </p>
        <p className="font-main text-white text-xs md:text-sm">
          김영일, 서영광, 배지아, 장나영
        </p>
     
      </div>
    </div>
  );
};

export default Footer;
