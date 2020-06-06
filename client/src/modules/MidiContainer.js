import React, { useRef, useEffect, useMemo, useState } from "react";
import MIDISounds from "midi-sounds-react";
import { observer } from "mobx-react-lite";
import { Wrapper, NoneWrapper } from "@/components/wrappers";
import PlayerContainer from "./player/PlayerContainer";
import InputContainer from "./input/InputContainer";
import { useStore } from "@/mobx";
import { useKeyPress } from "@/utils/hooks";

const INSTRUMENT = 834; // pre-defined instrument

/**
 * Main smart component of our application in which we inject our store
 */
const MidiContainer = () => {
  const { rollStore } = useStore();
  const midiRef = useRef(null);
  const isPlaying = useKeyPress(" ");
  const [, setInitialized] = useState(false);

  useEffect(() => {
    // Re-rendering our app for enabling the MIDI component
    setInitialized(true);
  }, []);

  const memoisedSong = useMemo(() => {
    // mapping request data to our MIDI format
    const songRawBeats = rollStore.pianoRollsSnapshot.map(([roll, x, y]) => [
      roll * 96 + x,
      y + 16
    ]);

    // creating empty beats
    const songBeats = Array.from(Array((16 * 96) / 8), () => [
      [],
      [[INSTRUMENT, [], 1 / 16]]
    ]);

    // adding notes to corresponding beats
    songRawBeats.forEach(([beat, pitch]) => {
      songBeats[(beat / 8).toFixed(0)][1][0][1] = [
        ...songBeats[(beat / 8).toFixed(0)][1][0][1],
        pitch
      ];
    });

    return songBeats;
  }, [rollStore.pianoRollsSnapshot]);

  /*
   * Watching for `isPlaying` prop so we start playing the beat accordingly
   */
  useEffect(() => {
    if (isPlaying) {
      midiRef.current.setEchoLevel(0.2);
      midiRef.current.setMasterVolume(rollStore.volume / 100);
      midiRef.current.startPlayLoop(
        memoisedSong,
        120,
        1 / 16,
        midiRef.current.beatIndex
      );
    } else {
      if (midiRef.current) {
        midiRef.current.stopPlayLoop();
        midiRef.current.beatIndex = 0;
      }
    }
  }, [isPlaying, memoisedSong, rollStore.volume]);

  // updating master volume
  useEffect(() => {
    if (midiRef.current) {
      midiRef.current.setMasterVolume(rollStore.volume / 100);
    }
  }, [rollStore.volume]);

  return (
    <Wrapper>
      <PlayerContainer
        isPlaying={isPlaying}
        rolls={rollStore.pianoRollsSnapshot}
        beatIndex={midiRef.current?.beatIndex}
      />
      <InputContainer
        setFeature={rollStore.setFeature}
        features={rollStore.featuresSnapshot}
        threshold={rollStore.threshold}
        setField={rollStore.setField}
        arousal={rollStore.arousal}
        valence={rollStore.valence}
        getSong={rollStore.getSong}
        volume={rollStore.volume}
        multiplier={rollStore.multiplier}
      />
      <NoneWrapper>
        <MIDISounds
          ref={midiRef}
          appElementName="root"
          instruments={[INSTRUMENT]}
        />
      </NoneWrapper>
    </Wrapper>
  );
};

export default observer(MidiContainer);
