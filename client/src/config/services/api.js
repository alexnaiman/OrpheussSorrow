import apisauce from "apisauce";
import getCookie from "../cookieProvider";
import apiCalls, { apiCallNames } from "./apiCalls";
import browserHistory from "../history";

const cookies = getCookie();

// our "constructor"
// http://localhost:8080/ is the address of the flask server
const create = (baseURL = "http://localhost:5000/") => {
  const api = apisauce.create({
    // base URL is read from the "constructor"
    baseURL,
    // here are some default headers
    headers: {},
    // 10 second timeout...
    timeout: 10240
  });

  // added here for future development
  // api.addResponseTransform(response => {
  //   if (response.status === 401) {
  //     browserHistory.push("/login");
  //   }
  // });
  // api.addRequestTransform(request => {
  //   const token = cookies.get("access_token");
  //   if (cookies.get("access_token")) {
  //     request.headers.Authentication = `Bearer ${token}`;
  //   }
  // });

  const calls = Object.keys(apiCalls).map(getApiCalls =>
    apiCalls[getApiCalls](api)
  );
  return {
    ...calls.reduce((a, b) => ({ ...a, ...b }))
  };
};

export default {
  create,
  callNames: apiCallNames
};
