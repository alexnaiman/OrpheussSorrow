import React, { memo } from "react";
import { NOTE_WIDTH, NOTE_HEIGHT, SPACING, HEIGHT } from "@/utils/constants";
import styled, { keyframes } from "styled-components";

/**
 * Dumb component that positions a note
 */
const NoteContainer = styled.div.attrs(({ x, y, roll }) => ({
  style: {
    top: HEIGHT - y * (NOTE_HEIGHT + SPACING) + SPACING / 2 + "px",
    left: (roll * 96 + x) * (NOTE_WIDTH + SPACING) + SPACING / 2
  }
}))`
  position: absolute;
`;

const fadeIn = keyframes`
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
`;

/**
 * Dumb component that renders a note
 */
const Note = styled.div`
  background: #100d54;
  width: ${NOTE_WIDTH + "px"};
  height: ${NOTE_HEIGHT + "px"};
  border: 2px solid #69f9fa;
  -webkit-box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 1);
  -moz-box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 1);
  box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 1);
  animation: ${fadeIn} 0.3s;
`;

const NoteFader = ({ x, y, roll }) => {
  return (
    <NoteContainer x={x} y={y} roll={roll}>
      <Note />
    </NoteContainer>
  );
};

export default memo(NoteFader);
