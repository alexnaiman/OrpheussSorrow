import React, { useState, useCallback, useRef, useEffect } from "react";
import { Grid, PianoRoll, Keys } from "@/components";
import {
  PlayerWrapper,
  RollWrapper,
  SliderWrapper
} from "@/components/player/playerWrappers";
import { throttle } from "lodash";

/**
 * Dumb Component that deals with rendering the player: keys, grid and notes
 * Also has a scroll spy that ensures that the `Keys` component position is updated accordingly
 * @param {array} rolls => the rolls of notes we render
 * @param {boolean} isPlaying => whether is playing or not the MIDI, used for translation animation
 */
const PlayerContainer = ({ rolls = [], isPlaying = false }) => {
  // our scrollSpy for the keys container
  const [scrollLeft, setScrollLeft] = useState(0);

  const scrollRef = useRef();
  const onScrollLeft = useCallback(
    throttle(e => {
      const {
        target: { scrollLeft: newScrollLeft }
      } = e;
      setScrollLeft(newScrollLeft);
    }, 200),
    []
  );

  useEffect(() => {
    const current = scrollRef.current;
    current.addEventListener("scroll", onScrollLeft);
    return () => {
      current.removeEventListener("scroll", onScrollLeft);
    };
  }, [onScrollLeft]);

  return (
    <PlayerWrapper
      ref={scrollRef}
      scrollLeft={scrollLeft}
      isPlaying={isPlaying}
    >
      <Keys scrollLeft={scrollLeft} />
      <SliderWrapper isPlaying={isPlaying}>
        <RollWrapper>
          <Grid />
          <PianoRoll rolls={rolls} />
        </RollWrapper>
      </SliderWrapper>
    </PlayerWrapper>
  );
};

export default PlayerContainer;
