import React, { memo } from "react";
import Slider from "react-input-slider";
import styled from "styled-components";

const SliderWrapper = styled.div``;

/**
 * custom design controlled slider component
 * @param {number} value -> the value of the slider
 * @param {string} primaryColor -> accent primary color -> used for customizing
 * @param {string} secondaryColor -> background secondary color -> used for customizing
 * @param {(value: number) => void} onChange -> for changing the value of our slider
 * @param {() => void} onDragEnd -> callback used when drag ends -> used for calls
 * @param {any} rest -> other props passed to the `react-input-slider` component for better customization
 */
const NeonSlider = ({
  value = 0,
  primaryColor = "rgba(105, 249, 250, 1)",
  secondaryColor = "#100d54",
  onChange = () => {},
  onDragEnd = () => {},
  ...rest
}) => {
  return (
    <SliderWrapper>
      <Slider
        axis="y"
        styles={{
          track: {
            backgroundColor: secondaryColor,
            width: 20,
            height: "100%",
            margin: "0 10px",
            borderRadius: 10,
            boxShadow: `0px 0px 5px 2px ${primaryColor}`
          },
          active: {
            backgroundColor: primaryColor,
            borderRadius: 10,
            boxShadow: `0px 0px 5px 2px ${primaryColor}`
          },
          thumb: {
            backgroundColor: secondaryColor,
            width: 30,
            height: 30,
            boxShadow: `0px 0px 5px 2px ${primaryColor}`
          }
        }}
        y={value}
        onChange={onChange}
        yreverse
        ymax={100}
        ymin={0}
        onDragEnd={onDragEnd}
        {...rest}
      />
    </SliderWrapper>
  );
};

export default memo(
  NeonSlider,
  (prevProps, nextProps) => prevProps.value === nextProps.value
);
