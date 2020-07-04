import React, { useCallback } from "react";
import { debounce } from "lodash";
import { InputWrapper, TextWrapper, Text } from "@/components/input/wrappers";
import { NeonSlider, CartesianSlider } from "@/components";

/**
 *  Dumb component that deals with rendering all our inputs
 * @param {Array<number>} features -> values for our features
 * @param {number} threshold -> threshold value for our model
 * @param {number} arousal -> arousal value for our model
 * @param {number} valence -> valence value for our model
 * @param {number} volume -> volume value for our MIDI player
 * @param {number} volume -> multiplier value for our MIDI player
 * @param {( index: number, value: number) => void} setFeature -> setter function for our features
 * @param {( field: string, value: any) => void} setField -> setter function for our model
 * @param {() => void} getSong -> request that gets a song for the given values
 */
const InputContainer = ({
  features,
  setFeature,
  threshold,
  setField,
  arousal,
  valence,
  getSong,
  volume,
  multiplier
}) => {
  const setThreshold = useCallback(
    ({ y }) => {
      setField("threshold", y);
    },
    [setField]
  );
  const setMultiplier = useCallback(
    ({ y }) => {
      setField("multiplier", y);
    },
    [setField]
  );

  const setVolume = useCallback(
    ({ y }) => {
      setField("volume", y);
    },
    [setField]
  );

  const setEmotionValue = useCallback(
    ({ x, y }) => {
      setField("arousal", y);
      setField("valence", x);
    },
    [setField]
  );

  const onDragEnd = useCallback(
    debounce(() => {
      getSong();
    }, 300),
    [getSong]
  );

  return (
    <InputWrapper>
      <CartesianSlider
        onDragEnd={onDragEnd}
        onChange={setEmotionValue}
        y={arousal}
        x={valence}
      />
      <TextWrapper>
        <Text primaryColor="#FEFF01">Threshold</Text>
      </TextWrapper>
      <NeonSlider
        primaryColor="#FEFF01"
        onChange={setThreshold}
        value={threshold}
        onDragEnd={onDragEnd}
        ymin={1}
      />
      <TextWrapper>
        <Text primaryColor="#FEFF01">Volume</Text>
      </TextWrapper>
      <NeonSlider primaryColor="#FEFF01" onChange={setVolume} value={volume} />
      <TextWrapper>
        <Text primaryColor="#FEFF01">Multiplier</Text>
      </TextWrapper>
      <NeonSlider
        primaryColor="#FEFF01"
        onChange={setMultiplier}
        value={multiplier}
        ymin={1}
        ymax={5}
        onDragEnd={onDragEnd}
      />
      <TextWrapper>
        <Text>Features</Text>
      </TextWrapper>
      {features.map((feature, index) => (
        <NeonSlider
          key={index}
          onDragEnd={onDragEnd}
          onChange={value => {
            setFeature(index, value);
          }}
          index={index}
          setFeature={setFeature}
          value={feature.value}
        />
      ))}
    </InputWrapper>
  );
};

export default InputContainer;
