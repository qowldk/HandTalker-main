import { Link } from "react-router-dom";
import { useSetRecoilState } from "recoil";
import { translateState } from "../../utils/recoil/atom";
import { HeaderButtonProps } from "../../types/HeaderButtonProps";

const HeaderButton = ({ isClicked, name, link }: HeaderButtonProps) => {
  const setTranslate = useSetRecoilState(translateState);
  const onClickListener = () => {
    setTranslate(false);
  };

  if (isClicked) {
    return (
      <div
        className={`w-20 border-b-2 border-black font-bold 
       mb-[-3px] h-full font-main text-base text-white flex flex-col justify-center items-center`}
        onClick={onClickListener}
      >
        <Link to={`${link}`} className="text-black"> {/* 색상 변경 */}
          <p>{name}</p>
        </Link>
      </div>
    );
  } else {
    return (
      <div
        className={`w-20 
       h-full font-main text-base text-white flex flex-col justify-center items-center`}
        onClick={onClickListener}
      >
        <Link to={`${link}`} className="text-black"> {/* 색상 변경 */}
          <p>{name}</p>
        </Link>
      </div>
    );
  }
};

export default HeaderButton;
