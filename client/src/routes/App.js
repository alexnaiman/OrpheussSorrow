import React, { Component } from "react";
import { Switch, Route } from "react-router-dom";

import { AppWrapper } from "../components";
import { MidiContainer } from "@/modules/index";

/**
 * Router structure - added for future development
 */
class App extends Component {
  render() {
    return (
      <AppWrapper>
        <Switch>
          <Route path="/" component={MidiContainer} />
        </Switch>
      </AppWrapper>
    );
  }
}
export default App;
