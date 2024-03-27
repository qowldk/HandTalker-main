import { atom } from "recoil";
import { recoilPersist } from "recoil-persist";

const { persistAtom } = recoilPersist();

export const translateState = atom({
  key: "translateState",
  default: false,
});

export const authState = atom({
  key: "authState",
  default: false,
  effects_UNSTABLE: [persistAtom],
});

export const resultText = atom({
  key: "resultText",
  default: "",
});

export const isGeneratingSentence = atom({  // LLM API 응답 대기중 여부
  key: "isGeneratingSentence",
  default: false,
});

export const dchannel = atom({
  key: "dchannel",
  default: "",
});
