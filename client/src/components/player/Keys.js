import React from "react";
import { KeysWrapper } from "./playerWrappers";
import { NOTE_RANGE, NOTE_HEIGHT, SPACING } from "@/utils/constants";
import styled from "styled-components";

/**
 * Presentational component that renders a key
 */
const Key = styled.div.attrs(({ isSharp }) => ({
  style: {
    background: isSharp ? "#69f9fa" : "#100d54"
  }
}))`
  height: ${NOTE_HEIGHT + SPACING - 2}px;
  font-size: ${NOTE_HEIGHT - 1}px;
  width: 100%;
  background: #100d54;
  color: white;
  display: flex;
  justify-content: flex-end;
  margin: 2px 0 2px;
  -webkit-box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 0.8);
  -moz-box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 0.8);
  box-shadow: 0px 0px 5px 2px rgba(105, 249, 250, 0.8);
`;

/**
 * Presentational component that renders all keys and positions them
 * @param {number} scrollLeft -> value of the scrollLeft position of the Container
 *                            -> used for dynamically changing the position of the keys onScroll
 */
const Keys = ({ scrollLeft }) => {
  return (
    <KeysWrapper scrollLeft={scrollLeft}>
      {NOTE_RANGE.map(item => (
        <Key key={item} isSharp={item.indexOf("#") > -1} />
      ))}
    </KeysWrapper>
  );
};

export default Keys;
