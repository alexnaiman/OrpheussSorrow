import styled from "styled-components";

/**
 * Wrapper components used only for styling and positioning our main content
 */

export const InputWrapper = styled.div`
  display: flex;
  height: 30vh;
  padding: 20px;
  -webkit-box-shadow: 0px -5px 19px -2px rgba(0, 0, 0, 0.75);
  -moz-box-shadow: 0px -5px 19px -2px rgba(0, 0, 0, 0.75);
  box-shadow: 0px -5px 19px -2px rgba(0, 0, 0, 0.75);
  z-index: 2;
  overflow-x: scroll;
  /* align-items: center; */
`;

export const Text = styled.span`
  transform: rotate(270deg);
  text-shadow: 0px 0px 7px
      ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 9px ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 12px
      ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 0px #000000, 0px 0px 0px #000000;
  color: #ffffff;
  letter-spacing: 0.5px;
  font-family: "Montserrat", sans-serif;
`;

export const TextWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  align-content: center;
  width: 20px;
`;

export const NeonInput = styled.input`
  background: transparent;
  padding: 10px;
  text-shadow: 0px 0px 7px
      ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 9px ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 12px
      ${({ primaryColor = "rgba(105, 249, 250, 1)" }) => primaryColor},
    0px 0px 0px #000000, 0px 0px 0px #000000;
  color: #ffffff;
  letter-spacing: 0.5px;
  font-family: "Montserrat", sans-serif;
`;

export const SliderWrapper = styled.div`
  display: flex;
`;

export const CartesianTextWrapper = styled.div`
  justify-content: center;
  transform: initial;
  padding-top: 10px;
`;

export const LeftWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;
