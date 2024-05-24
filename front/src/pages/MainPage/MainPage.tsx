import React from "react";
import Header from "../../components/Header/Header";
import Page1 from "./Page1";
import Page2 from "./Page2";
import Page3 from "./Page3";
import Page4 from "./Page4";
import Page5 from "./Page5";
import { SectionsContainer, Section } from "react-fullpage";
import "./MainPage.css"; // CSS 파일을 import

let options = {
  activeClass: "active", // the class that is appended to the sections links
  anchors: [
    "sectionOne",
    "sectionTwo",
    "sectionThree",
    "sectionFour",
    "sectionFive",
  ], // the anchors for each sections
  arrowNavigation: true, // use arrow keys
  className: "SectionContainer", // the class name for the section container
  delay: 500, // the scroll animation speed
  navigation: false, // use dots navigation
  scrollBar: false, // use the browser default scrollbar
  sectionClassName: "Section", // the section class name
  sectionPaddingTop: "0", // the section top padding
  sectionPaddingBottom: "0", // the section bottom padding
  verticalAlign: false, // align the content of each section vertical
};

const MainPage = () => {
  return (
    <div className="flex">
      <Header />
      <SectionsContainer {...options}>
        <Section>
          <Page1 />
        </Section>
        <Section>
          <Page2 />
        </Section>
        <Section>
          <Page3 />
        </Section>
    
    
      </SectionsContainer>
    </div>
  );
};

export default MainPage;
