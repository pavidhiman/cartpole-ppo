import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../Api';

export const getUser = createAsyncThunk('user/getUser', async (args) => {
  const { userID } = args;

  try {
    const response = await api.get(`/Patient/GetUser/${userID}`);
    return response.data;
  } catch (error) {
    console.error('Error while getting user ', error);
    throw error;
  }
});

export const updateUserTypeID = createAsyncThunk('user/updateUserTypeID', async (args) => {
  const { userID } = args;

  try {
    // hard coding user type id as 1 (patient) for now
    const response = await api.put(`/Patient/updateUserTypeId/${userID}/1`, {});
    return response.data;
  } catch (error) {
    console.error('Error while updating user type id ', error);
    throw error;
  }
});

export const setUserSetupCompleted = createAsyncThunk('user/setUserSetupCompleted', async (args) => {
  const { userID, setupCompleted } = args;

  try {
    const response = await api.put(`/Patient/SetUserSetupCompleted/${userID}/${setupCompleted}`, {});
    return response.data;
  } catch (error) {
    console.error('Error while setting UserSetupCompleted ', error);
    throw error;
  }
});

export const updateUserAccountStatus = createAsyncThunk('user/updateUserAccountStatus', async (args) => {
  const { userID, isPWP, hasCondition, suspectedCondition, isTester } = args;

  try {
    const response = await api.put(
      `/Patient/UpdateUserAccountStatus/${userID}/${isPWP}/${hasCondition}/${suspectedCondition}/${isTester}`,
      {},
    );
    return response.data;
  } catch (error) {
    console.error('Error while updating UserAccountStatus ', error);
    throw error;
  }
});

const userSlice = createSlice({
  name: 'user',
  initialState: {
    getUserData: null,
    updateUserTypeData: null,
    setUserSetupCompletedData: null,
    updateUserAccountStatusData: null,
    userSetupStartTime: null,
    getUserStatus: 'idle',
    updateUserTypeStatus: 'idle',
    setUserSetupCompletedStatus: 'idle',
    updateUserAccountStatus: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(getUser.pending, (state) => {
        state.getUserStatus = 'loading';
      })
      .addCase(getUser.fulfilled, (state, action) => {
        state.getUserStatus = 'succeeded';
        state.getUserData = action.payload;
      })
      .addCase(getUser.rejected, (state, action) => {
        state.getUserStatus = 'failed';
        state.error = action.error.message;
      })
      .addCase(updateUserTypeID.pending, (state) => {
        state.updateUserTypeStatus = 'loading';
      })
      .addCase(updateUserTypeID.fulfilled, (state, action) => {
        state.updateUserTypeStatus = 'succeeded';
        state.updateUserTypeData = action.payload;
      })
      .addCase(updateUserTypeID.rejected, (state, action) => {
        state.updateUserTypeStatus = 'failed';
        state.error = action.error.message;
      })
      .addCase(setUserSetupCompleted.pending, (state) => {
        state.setUserSetupCompletedStatus = 'loading';
      })
      .addCase(setUserSetupCompleted.fulfilled, (state, action) => {
        state.setUserSetupCompletedStatus = 'succeeded';
        state.setUserSetupCompletedData = action.payload;
      })
      .addCase(setUserSetupCompleted.rejected, (state, action) => {
        state.setUserSetupCompletedStatus = 'failed';
        state.error = action.error.message;
      })
      .addCase(updateUserAccountStatus.pending, (state) => {
        state.updateUserAccountStatus = 'loading';
      })
      .addCase(updateUserAccountStatus.fulfilled, (state, action) => {
        state.updateUserAccountStatus = 'succeeded';
        state.updateUserAccountStatusData = action.payload;
      })
      .addCase(updateUserAccountStatus.rejected, (state, action) => {
        state.updateUserAccountStatus = 'failed';
        state.error = action.error.message;
      });
  },
});

export default userSlice.reducer;
