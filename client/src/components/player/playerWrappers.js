import styled, { keyframes } from "styled-components";
import { KEYS_WIDTH, WIDTH } from "@/utils/constants";

/**
 * Wrapper components used only for styling and positioning our main content
 */

export const RollWrapper = styled.div`
  background-color: rgba(0, 0, 0, 0.2);
  position: relative;
  margin-left: ${KEYS_WIDTH}px;
  transition: transform linear 0.5s;
`;

const playingAnim = keyframes`
  from {
    transform: translateX(0);
  }

  to {
    transform: translateX(-${WIDTH}px);
  }
`;

export const SliderWrapper = styled.div`
  width: ${1000}px;
  animation: ${({ isPlaying }) => isPlaying && playingAnim} ${24}s linear
    infinite;
`;

export const PlayerWrapper = styled.div`
  overflow: scroll;
  position: relative;
  height: 70vh;
  display: flex;
  /* ::-webkit-scrollbar {
    display: none; // TODO: SEE IF THE BUG PERSISTS
    /* opacity: 0
   */
`;

export const KeysWrapper = styled.div.attrs(({ scrollLeft }) => ({
  style: {
    transform: `translateX(${scrollLeft}px)`,
    WebkitTransform: `translateX(${scrollLeft}px)`
  }
}))`
  height: 100%;
  width: ${KEYS_WIDTH}px;
  position: absolute;
  z-index: 1;
  will-change: transform;
  transition: transform 0.5s linear;
`;
