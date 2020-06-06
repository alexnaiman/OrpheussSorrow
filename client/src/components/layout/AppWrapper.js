import styled from "styled-components";

/**
 * App wrapper component used only for styling and positioning our main content
 */
export default styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  opacity: 1 !important;
  position: relative;
  transition: opacity 0.5s;
  background: #e3257d; /* fallback for old browsers */
  background: -webkit-linear-gradient(
    to right,
    #2a1759,
    #e3257d
  ); /* Chrome 10-25, Safari 5.1-6 */
  background: linear-gradient(
    to right,
    #2a1759,
    #e3257d
  ); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
  /* overflow: hidden; */
`;
