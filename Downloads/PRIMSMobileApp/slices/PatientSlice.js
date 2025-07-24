import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';
import api from '../Api';
import patientData from '../data/patient.json';
const USE_LOCAL_DATA = true;

export const getPatient = createAsyncThunk('patient/getPatient', async (args) => {
  const { patientID } = args;

  if (USE_LOCAL_DATA) {
    return patientData;
  }

  try {
    const response = await api.get(`/Patient/GetPatient/${patientID}`);
    return response.data;
  } catch (error) {
    console.error('Error while getting patient ', error);
    throw error;
  }
});

const userSlice = createSlice({
  name: 'patient',
  initialState: {
    getPatientData: null,
    getPatientStatus: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(getPatient.pending, (state) => {
        state.getPatientStatus = 'loading';
      })
      .addCase(getPatient.fulfilled, (state, action) => {
        state.getPatientStatus = 'succeeded';
        state.getPatientData = action.payload;
      })
      .addCase(getPatient.rejected, (state, action) => {
        state.getPatientStatus = 'failed';
        state.error = action.error.message;
      });
  },
});

export default userSlice.reducer;
