import logo from "../../assets/icons/logo.svg";
import sl from "../../assets/icons/SignLanguage.png"; // 이미지 경로 수정
import menu from "../../assets/icons/menu.svg";
import HeaderButton from "./HeaderButton";
import { useLocation } from "react-router-dom";
import { useRecoilState } from "recoil";
import { authState } from "../../utils/recoil/atom";
import { Link } from "react-router-dom";
import Swal from "sweetalert2";
import { useState } from "react";
import { Drawer } from "../Drawer/Drawer";

const Header = () => {
  const location = useLocation();
  const path = location.pathname;
  const [auth, setAuth] = useRecoilState(authState);

  const [open, setOpen] = useState(false);
  const openHandler = () => {
    setOpen(!open);
  };

  // const Toast = Swal.mixin({
  //   toast: true,
  //   position: "top-right",
  //   showConfirmButton: false,
  //   timer: 1500,
  //   timerProgressBar: true,
  //   didOpen: (toast) => {
  //     toast.addEventListener("mouseenter", Swal.stopTimer);
  //     toast.addEventListener("mouseleave", Swal.resumeTimer);
  //   },
  // });



  return (
    <>
      {open && <Drawer onOpen={openHandler} isOpen={open} />}
     
      <div className="hidden lg:flex bg-navy bg-opacity-70 top-0 fixed z-10 w-full h-[4.5rem] flex-row justify-between items-center">
        <div className="ml-[10rem]">
          <Link to="/" className="flex items-center justify-center w-full"> {/* 이미지를 감싸는 div 요소에 flex와 justify-center 클래스 추가 */}
            <img src={sl} alt="sl" className="w-auto h-16 items-center justify-center mx-auto" /> {/* 이미지 크기 조정 및 수평 가운데 정렬 */}
          </Link>
        </div>

        <div className="flex flex-row items-center justify-center h-full mr-[10vw]">
          <HeaderButton
            isClicked={path === "/" ? true : false}
            name="메인"
            link="/"
          />
          <HeaderButton
            isClicked={path === "/translate" ? true : false}
            name="번역"
            link="/translate"
          />
    
        </div>
      </div>
    </>
  );
};

export default Header;
