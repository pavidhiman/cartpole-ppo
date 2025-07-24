import * as Amplitude from '@amplitude/analytics-react-native';
import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import api from '../Api';
import { getAccessToken } from '../utils/Library';

export const getValidUser = createAsyncThunk('validUser/validateUser', async (args) => {
  const { email } = args;
  console.log('[getValidUser] Validating user with email:', email); // Debug log

  try {
    const token = await getAccessToken();
    const url = `/patient/validateuser/${email}`;
    console.log('[getValidUser] Request URL:', url); // Debug log
    const response = await api.get(
      url,
      { headers: { Authorization: `Bearer ${token}` } }
    );
    console.log('[getValidUser] API response:', response.data); // Debug log
    Amplitude.logEvent('VALIDATE_USER_RESPONSE', {
      'response': response.data,
    });
    return response.data;
  } catch (error) {
    console.error('[getValidUser] Error occurred while getting valid user', error);
    Amplitude.logEvent('VALIDATE_USER_ERROR', {
      'error': error,
      'email': email,
    });
    throw error;
  }
});

const validUserSlice = createSlice({
  name: 'validUser',
  initialState: {
    data: null,
    status: 'idle',
    error: null,
  },
  reducers: {
    updatePatientID: (state, action) => {
      if (state.data) {
        state.data.patientID = action.payload;
      }
    },
    setValidUserMock: (state, action) => {
      state.status = 'succeeded';
      state.data = action.payload;
    }    
  },
  extraReducers: (builder) => {
    builder
      .addCase(getValidUser.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(getValidUser.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.data = action.payload;
      })
      .addCase(getValidUser.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  },
});

export const { updatePatientID, setValidUserMock } = validUserSlice.actions;
export default validUserSlice.reducer;
