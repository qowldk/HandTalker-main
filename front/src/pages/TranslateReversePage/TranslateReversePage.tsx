import Footer from "../../components/Footer/Footer";
import Header from "../../components/Header/Header";
import TranslateReverse from "./translateReverse/TranslateReverse";

const TranslateReversePage = () => {
  return (
    <div className="flex flex-col h-[100vh] w-full">
      <Header />
      <TranslateReverse />
      <Footer />
    </div>
  );
};

export default TranslateReversePage;
