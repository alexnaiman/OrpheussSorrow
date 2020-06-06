import { useState, useEffect } from "react";
/**
 * Custom hook component used for watching a key press -> functions as a toggle
 * @param {string} targetKey Key we are watching for
 */
export function useKeyPress(targetKey) {
  // State for keeping track of whether key is pressed
  const [keyPressed, setKeyPressed] = useState(false);

  // If pressed key is our target key then set to true

  // Add event listeners
  useEffect(() => {
    function downHandler(e) {
      e.preventDefault();
      if (e.key === targetKey) {
        setKeyPressed(!keyPressed);
      }
    }

    // If released key is our target key then set to false

    window.addEventListener("keydown", downHandler);
    // Remove event listeners on cleanup
    return () => {
      window.removeEventListener("keydown", downHandler);
    };
  }, [keyPressed, targetKey]); // Empty array ensures that effect is only run on mount and unmount

  return keyPressed;
}
