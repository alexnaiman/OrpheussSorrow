const {
  override,
  disableEsLint,
  addDecoratorsLegacy,
  addBabelPlugins,
  fixBabelImports,
  addWebpackAlias
} = require("customize-cra");
const path = require("path");

module.exports = override(
  disableEsLint(),
  addDecoratorsLegacy(),
  ...addBabelPlugins(
    "babel-plugin-styled-components",
    "@babel/plugin-proposal-export-default-from"
  ),
  fixBabelImports("react-app-rewire-mobx", {
    libraryDirectory: "",
    camel2DashComponentName: false
  }),
  addWebpackAlias({
    "@": path.resolve(__dirname, "src")
  })
);
