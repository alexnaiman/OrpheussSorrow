import React from "react";
import ReactDOM from "react-dom";
import { createGlobalStyle } from "styled-components";
import { Router } from "react-router-dom";

import App from "@/routes/App";
import * as serviceWorker from "@/serviceWorker";
import reset from "@/css/reset";

import { StoreProvider, getStoreInstance } from "./mobx";
import browserHistory from "./config/history";

const GlobalStyle = createGlobalStyle`${reset}`;

const mainStore = getStoreInstance();

/**
 * Root of our app in which we inject our mobx context and router
 */
ReactDOM.render(
  <StoreProvider value={mainStore}>
    <>
      <Router history={browserHistory}>
        <App />
      </Router>
      <GlobalStyle />
    </>
  </StoreProvider>,
  document.getElementById("root")
);
// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
