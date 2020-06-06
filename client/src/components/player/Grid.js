import React, { memo } from "react";
import {
  NOTE_WIDTH,
  NOTE_HEIGHT,
  SPACING,
  HEIGHT,
  WIDTH
} from "@/utils/constants";
import styled from "styled-components";

const HORIZONTAL_LINES = Array(97).fill(1);
const VERTICAL_LINES = Array(16 * 96 + 2).fill(1);

const HorizontalLine = styled.div.attrs(({ lineNo }) => ({
  style: {
    background: "rgba(255, 255, 255, 0.2)",
    width: WIDTH + NOTE_WIDTH - SPACING + "px",
    height: "1px",
    position: "absolute",
    top: lineNo * (NOTE_HEIGHT + SPACING) + "px"
  }
}))``;

const VerticalLine = styled.div.attrs(({ lineNo }) => ({
  style: {
    background: "rgba(255, 255, 255, 0.2)",
    height: HEIGHT - 2 * SPACING + "px",
    width: "1px",
    position: "absolute",
    left: lineNo * (NOTE_WIDTH + SPACING) + "px"
  }
}))``;

/**
 * Grid component that deals with rendering all the lines and columns
 */
const Grid = () => {
  return (
    <div>
      <div>
        {HORIZONTAL_LINES.map((_, index) => (
          <HorizontalLine lineNo={index} key={`row${index}`} />
        ))}
      </div>
      <div>
        {VERTICAL_LINES.map((_, index) => (
          <VerticalLine lineNo={index} key={`column${index}`} />
        ))}
      </div>
    </div>
  );
};

export default memo(Grid, () => true);
