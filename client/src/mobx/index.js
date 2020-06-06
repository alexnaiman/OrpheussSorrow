import { types } from "mobx-state-tree";
import { stores } from "./stores";
import { createContext, useContext } from "react";
import api from "@/config/services/api";

const apiService = api.create();

const mainStore = types.model("MainStore", {
  ...stores
});

let instance = null;

export function getStoreInstance() {
  if (!instance) {
    instance = mainStore.create({}, { apiService, callNames: api.callNames });
  }

  return instance;
}

export const StoreContext = createContext({});

export const useStore = () => useContext(StoreContext);

export const StoreProvider = StoreContext.Provider;
