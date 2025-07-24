import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../Api';

export const getOneYrAssessments = createAsyncThunk('assessment/getOneYrAssessments', async (args) => {
  const { patientID } = args;

  try {
    const response = await api.get(`/Assessment/getOneYrAssessments/${patientID}`);
    return response.data;
  } catch (error) {
    console.error('Error in getOneYrAssessments (AssessmentSlice)', error);
    throw error;
  }
});

const assessmentSlice = createSlice({
  name: 'assessments',
  initialState: {
    getOneYrAssessmentsData: [],
    getOneYrAssessmentsStatus: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      //getOneYrAssessments
      .addCase(getOneYrAssessments.pending, (state) => {
        state.getOneYrAssessmentsStatus = 'loading';
      })
      .addCase(getOneYrAssessments.fulfilled, (state, action) => {
        state.getOneYrAssessmentsStatus = 'succeeded';
        state.getOneYrAssessmentsData = action.payload;
      })
      .addCase(getOneYrAssessments.rejected, (state, action) => {
        state.getOneYrAssessmentsStatus = 'failed';
        state.error = action.error.message;
      });
  },
});

export default assessmentSlice.reducer;
