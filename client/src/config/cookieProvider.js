import Cookies from "universal-cookie";

let instance = null;

export default function getApi() {
  if (!instance) {
    instance = new Cookies();
  }
  return instance;
}
