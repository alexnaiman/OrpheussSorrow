// Here we will make our api calls regarding our rolls
export default function geRollApiCalls(api) {
  return {
    getSong: body => api.post("/get_song", body)
  };
}
// here we export each name for each api call - maybe we will generate those
export const callNames = {
  GET_SONG: "getSong"
};
