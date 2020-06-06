import React, { memo } from "react";
import shortId from "short-id";
import Note from "./Note";
import "./pianoRoll.scss";

/**
 *  Dumb component that renders a roll of notes
 * @param {Array<Array<number>>} rolls -> an array representing our main notes represented as a sparse matrix with
 * (roll, x, y) representing the coordinates in time and "tone" of our note
 */
const PianoRoll = ({ rolls = [] }) => {
  return rolls.map(([roll, x, y]) => (
    <Note roll={roll} x={x} y={y} value={1} key={shortId.generate()} />
  ));
};

export default memo(PianoRoll);
