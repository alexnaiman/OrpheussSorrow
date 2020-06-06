import { types } from "mobx-state-tree";

export const Feature = types.model("Feature", {
  value: types.number
});
