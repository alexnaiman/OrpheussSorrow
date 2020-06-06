import { types, flow } from "mobx-state-tree";
import baseStore from "./base";
import { getEnv, getSnapshot } from "mobx-state-tree";
import { Roll, Feature } from "../models/index";
import { DEFAULT_FEATURES } from "@/utils/constants";

// Here we will add call our api service for requests *just like sagas/thunks*
const pianoRollsStore = types
  .model("Roll", {
    threshold: types.optional(types.number, 25),
    multiplier: types.optional(types.number, 1),
    volume: types.optional(types.number, 25),
    arousal: types.optional(types.number, 50),
    valence: types.optional(types.number, 50),
    pianoRolls: types.array(Roll),
    features: types.optional(types.array(Feature), DEFAULT_FEATURES)
  })
  .views(self => ({
    get featuresSnapshot() {
      return getSnapshot(self.features);
    },
    get pianoRollsSnapshot() {
      return getSnapshot(self.pianoRolls);
    }
  }))
  .actions(self => ({
    afterCreate: () => {
      // get random song on start-up
      self.getSong();
    },
    setField: (field, value) => {
      self[field] = value;
    },
    setFeature: (index, value) => {
      self.features[index] = { value: value.y };
    },
    getSong: flow(function* getSongRequest() {
      if (self.isLoading) return;
      const calls = getEnv(self).callNames.rollCallNames;
      yield self.fetch(
        calls.GET_SONG,
        {
          features: getSnapshot(self.features).map(item => item.value),
          threshold: self.threshold,
          valence: self.valence * self.multiplier,
          arousal: self.arousal * self.multiplier
        },
        self.onGetSongSuccess,
        self.onGetSongError
      );
    }),
    onGetSongSuccess: response => {
      self.setField("pianoRolls", response);
    },
    onGetSongError: errorResponse => {
      console.log(errorResponse);
    }
  }));

const enhancedAuth = types.compose(
  pianoRollsStore,
  baseStore
);
export default enhancedAuth;
