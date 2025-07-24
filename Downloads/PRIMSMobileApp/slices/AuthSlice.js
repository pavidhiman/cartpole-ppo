import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import api from '../Api';
import { getAccessToken, getRefreshToken } from '../utils/Library';
export const fetchAccessToken = createAsyncThunk('auth/fetchAccessToken', async () => {
  const token = await getAccessToken();
  if (!token) {
    throw new Error('Access token not found');
  }
  return token;
});

export const fetchRefreshToken = createAsyncThunk('auth/fetchRefreshToken', async () => {
  const token = await getRefreshToken();
  if (!token) {
    throw new Error('Refresh token not found');
  }
  return token;
});

export const loginTrial = createAsyncThunk(
  'auth/loginTrial',
  async ({ trialId }, { rejectWithValue }) => {
    // Toggle this constant while backend isn’t ready
    const USE_LOCAL_DATA = true;
  

    try {
      if (USE_LOCAL_DATA) {
        const data = require('../data/mock_users.json');
        const user = data.find(
          (u) => u.userType === 'trial' && u.trialId === trialId
        );
        if (!user) throw new Error('Invalid Trial ID');
        return user;
      } else {
        const res = await api.post('/Auth/loginTrial', { trialId });
        return res.data;
      }
    } catch (err) {
      return rejectWithValue(err.message || 'Login failed');
    }
  }
);


const authSlice = createSlice({
  name: 'auth',
  initialState: {
    accessToken: null,
    refreshToken: null,
    accessTokenStatus: 'idle',
    refreshTokenStatus: 'idle',
    error: null,
    trialLoginStatus: 'idle',
    user: null, 
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchAccessToken.pending, (state) => {
        state.accessTokenStatus = 'loading';
      })
      .addCase(fetchAccessToken.fulfilled, (state, action) => {
        state.accessTokenStatus = 'succeeded';
        state.accessToken = action.payload;
      })
      .addCase(fetchAccessToken.rejected, (state, action) => {
        state.accessTokenStatus = 'failed';
        state.error = action.error.message;
      })
      .addCase(fetchRefreshToken.pending, (state) => {
        state.refreshTokenStatus = 'loading';
      })
      .addCase(fetchRefreshToken.fulfilled, (state, action) => {
        state.refreshTokenStatus = 'succeeded';
        state.refreshToken = action.payload;
      })
      .addCase(fetchRefreshToken.rejected, (state, action) => {
        state.refreshTokenStatus = 'failed';
        state.error = action.error.message;
      })
      .addCase(loginTrial.pending, (state) => {
        state.trialLoginStatus = 'loading';
      })
      .addCase(loginTrial.fulfilled, (state, action) => {
        state.trialLoginStatus = 'succeeded';
        state.user = action.payload;          // userType === 'trial'
      })
      .addCase(loginTrial.rejected, (state, action) => {
        state.trialLoginStatus = 'failed';
        state.error = action.payload;
      })
        
  },
});

export default authSlice.reducer;
