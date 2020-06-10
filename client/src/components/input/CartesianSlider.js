import React, { memo } from "react";
import Slider from "react-input-slider";
import {
  Text,
  TextWrapper,
  SliderWrapper,
  LeftWrapper,
  CartesianTextWrapper
} from "./wrappers";

/**
 * Two-axis slider usd for our emtional values
 * @param {number} x -> x-axes value
 * @param {number} y -> y-axes value
 * @param {({x: number, y: number}) => value} y -> `onChange` function for the `react-input-slider`
 * @param {() => void} onDragEnd -> callback used when drag ends -> used for calls
 */
const CartesianSlider = ({ x, y, onChange, onDragEnd }) => {
  return (
    <SliderWrapper>
      <TextWrapper>
        <Text>Arousal</Text>
      </TextWrapper>
      <LeftWrapper>
        <Slider
          axis="xy"
          styles={{
            track: {
              background:
                "linear-gradient(to top left,  #e3257d, rgba(255, 153, 150, 0), rgba(105, 249, 250, 1)), linear-gradient(to top right, #100d54, rgba(255, 153, 150, 0), #2a1759) ;",
              height: "90%",
              margin: "0 10px",
              borderRadius: 0,
              boxShadow: `3px 3px 3px -1px #e3257d, -3px -3px 3px -1px rgba(105, 249, 250, 1)`,
              alignSelf: "center"
            },
            active: {
              borderRadius: 10
            },
            thumb: {
              backgroundColor: "#100d544f",
              boxShadow: `2px 2px 5px -1px #e3257d, -2px -2px 5px -1px rgba(105, 249, 250, 1)`
            }
          }}
          y={y}
          x={x}
          onChange={onChange}
          onDragEnd={onDragEnd}
          yreverse
          ymax={100}
          ymin={0}
        />
        <CartesianTextWrapper>
          <Text primaryColor="#e3257d">Valence</Text>
        </CartesianTextWrapper>
      </LeftWrapper>
    </SliderWrapper>
  );
};

export default memo(
  CartesianSlider,
  (prevProps, nextProps) =>
    prevProps.x === nextProps.x && prevProps.y === nextProps.y
);
