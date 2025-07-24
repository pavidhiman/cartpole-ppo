import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import api from '../Api';
import locationsData from '../data/locations.json'; // temp json for UI testing
const USE_LOCAL_DATA = true;

export const getLocations = createAsyncThunk('locations/getLocations', async () => {
  if (USE_LOCAL_DATA) {
    return locationsData;
  }
  try {
    const response = await api.get(`/Patient/GetLocations`);
    return response.data;
  } catch (error) {
    console.error('Error occurred while getting locations', error);
    throw error;
  }
});

const locationsSlice = createSlice({
  name: 'locations',
  initialState: {
    getLocationsData: null,
    getLocationsStatus: 'idle',
    error: null,
    selectedLocation: null,
  },
  reducers: {
    setSelectedLocationRedux: (state, action) => {
      state.selectedLocation = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getLocations.pending, (state) => {
        state.getLocationsStatus = 'loading';
      })
      .addCase(getLocations.fulfilled, (state, action) => {
        state.getLocationsStatus = 'succeeded';
        state.getLocationsData = action.payload;
      })
      .addCase(getLocations.rejected, (state, action) => {
        state.getLocationsStatus = 'failed';
        state.error = action.error.message;
      });
  },
});

export const { setSelectedLocationRedux } = locationsSlice.actions;

export default locationsSlice.reducer;
