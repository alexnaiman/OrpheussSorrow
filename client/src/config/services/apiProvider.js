import { create } from "./api";

let instance = null;

export default function getApi() {
  if (!instance) {
    instance = create();
  }
  return instance;
}
