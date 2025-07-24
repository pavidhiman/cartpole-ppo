import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../Api';
import pushTokenData from '../data/pushToken.json'; // temp json for UI testing

const USE_LOCAL_DATA = true;
export const addPushToken = createAsyncThunk('pushToken/addPushToken', async (args) => {
  const { data } = args;
  if (USE_LOCAL_DATA) {
    return pushTokenData;
  }
  try {
    const response = await api.post(`/Patient/AddUserPushToken`, data);
    return response.data;
  } catch (error) {
    console.error('Error occurred while saving push token', error);
    throw error;
  }
});

const pushTokenSlice = createSlice({
  name: 'pushToken',
  initialState: {
    data: null,
    status: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(addPushToken.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(addPushToken.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.data = action.payload;
      })
      .addCase(addPushToken.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  },
});

export default pushTokenSlice.reducer;
