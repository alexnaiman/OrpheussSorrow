import { types } from "mobx-state-tree";
import authStore from "./rollStore";

export const stores = {
  rollStore: types.optional(authStore, {})
};
