import styled from "styled-components";

/**
 * Wrapper components used only for styling and positioning our main content
 */

export const Wrapper = styled.div`
  flex: 1;
  position: relative;
`;

export const NoneWrapper = styled.div`
  opacity: 0;
  position: absolute;
  z-index: -1;
  height: 0;
`;
