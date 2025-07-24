import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../Api';

export const saveNewInvitation = createAsyncThunk('patient/saveNewInvitation', async (args) => {
  const { patientID, userTypeID, relationTypeID, userEmail } = args;

  try {
    const response = await api.get(
      `/patient/saveNewInvitation/${patientID}/${userTypeID}/${relationTypeID}/${userEmail}`,
    );
    return response.data;
  } catch (error) {
    console.error('Error in saveNewInvitation (InvitationsSlice)', error);
    throw error;
  }
});

export const acceptInvitation = createAsyncThunk('patient/acceptInvitation', async (args) => {
  const { patientCaregiverID, userID } = args;

  try {
    const response = await api.get(`/patient/acceptInvitation/${patientCaregiverID}/${userID}`);
    return response.data;
  } catch (error) {
    console.error('Error in acceptInvitation (InvitationsSlice)', error);
    throw error;
  }
});

export const getPendingInvites = createAsyncThunk('patient/getPendingInvites', async (args) => {
  const { patientID } = args;

  try {
    const response = await api.get(`/patient/getPendingInvites/${patientID}`);
    return response.data;
  } catch (error) {
    console.error('Error in getPendingInvites (InvitationsSlice)', error);
    throw error;
  }
});

export const getAcceptedInvites = createAsyncThunk('patient/getAcceptedInvites', async (args) => {
  const { patientID } = args;

  try {
    const response = await api.get(`/patient/getAcceptedInvites/${patientID}`);
    return response.data;
  } catch (error) {
    console.error('Error in getAcceptedInvites (InvitationsSlice)', error);
    throw error;
  }
});

export const cancelInvitation = createAsyncThunk('patient/cancelInvitation', async (args) => {
  const { patientID, patientCaregiverID, userID } = args;

  try {
    const response = await api.get(`/patient/cancelInvitation/${patientID}/${patientCaregiverID}/${userID}`);
    return response.data;
  } catch (error) {
    console.error('Error in cancelInvitation (InvitationsSlice)', error);
    throw error;
  }
});

export const getUserAcceptedInvites = createAsyncThunk('patient/getUserAcceptedInvites', async (args) => {
  const { userID } = args;

  try {
    const response = await api.get(`/patient/getUserAcceptedInvites/${userID}`);
    return response.data;
  } catch (error) {
    console.error('Error in getUserAcceptedInvites (InvitationsSlice)', error);
    throw error;
  }
});

export const getUserPendingInvites = createAsyncThunk('patient/getUserPendingInvites', async (args) => {
  const { userID } = args;

  try {
    const response = await api.get(`/patient/getUserPendingInvites/${userID}`);
    return response.data;
  } catch (error) {
    console.error('Error in getUserPendingInvites (InvitationsSlice)', error);
    throw error;
  }
});

const invitationsSlice = createSlice({
  name: 'invitations',
  initialState: {
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      // saveNewInvitation
      .addCase(saveNewInvitation.pending, (state) => {
        state.saveNewInvitation = 'loading';
      })
      .addCase(saveNewInvitation.fulfilled, (state, action) => {
        state.saveNewInvitation = 'succeeded';
        state.saveNewInvitation = action.payload;
      })
      .addCase(saveNewInvitation.rejected, (state, action) => {
        state.saveNewInvitation = 'failed';
        state.error = action.error.message;
      })
      // acceptInvitation
      .addCase(acceptInvitation.pending, (state) => {
        state.acceptInvitation = 'loading';
      })
      .addCase(acceptInvitation.fulfilled, (state, action) => {
        state.acceptInvitation = 'succeeded';
        state.acceptInvitation = action.payload;
      })
      .addCase(acceptInvitation.rejected, (state, action) => {
        state.acceptInvitation = 'failed';
        state.error = action.error.message;
      })
      // getPendingInvites
      .addCase(getPendingInvites.pending, (state) => {
        state.getPendingInvites = 'loading';
      })
      .addCase(getPendingInvites.fulfilled, (state, action) => {
        state.getPendingInvites = 'succeeded';
        state.getPendingInvites = action.payload;
      })
      .addCase(getPendingInvites.rejected, (state, action) => {
        state.getPendingInvites = 'failed';
        state.error = action.error.message;
      })
      // getAcceptedInvites
      .addCase(getAcceptedInvites.pending, (state) => {
        state.getAcceptedInvites = 'loading';
      })
      .addCase(getAcceptedInvites.fulfilled, (state, action) => {
        state.getAcceptedInvites = 'succeeded';
        state.getAcceptedInvites = action.payload;
      })
      .addCase(getAcceptedInvites.rejected, (state, action) => {
        state.getAcceptedInvites = 'failed';
        state.error = action.error.message;
      })
      // cancelInvitation
      .addCase(cancelInvitation.pending, (state) => {
        state.cancelInvitation = 'loading';
      })
      .addCase(cancelInvitation.fulfilled, (state, action) => {
        state.cancelInvitation = 'succeeded';
        state.cancelInvitation = action.payload;
      })
      .addCase(cancelInvitation.rejected, (state, action) => {
        state.cancelInvitation = 'failed';
        state.error = action.error.message;
      })
      // getUserAcceptedInvites
      .addCase(getUserAcceptedInvites.pending, (state) => {
        state.getUserAcceptedInvites = 'loading';
      })
      .addCase(getUserAcceptedInvites.fulfilled, (state, action) => {
        state.getUserAcceptedInvites = 'succeeded';
        state.getUserAcceptedInvites = action.payload;
      })
      .addCase(getUserAcceptedInvites.rejected, (state, action) => {
        state.getUserAcceptedInvites = 'failed';
        state.error = action.error.message;
      })
      // getUserPendingInvites
      .addCase(getUserPendingInvites.pending, (state) => {
        state.getUserPendingInvites = 'loading';
      })
      .addCase(getUserPendingInvites.fulfilled, (state, action) => {
        state.getUserPendingInvites = 'succeeded';
        state.getUserPendingInvites = action.payload;
      })
      .addCase(getUserPendingInvites.rejected, (state, action) => {
        state.getUserPendingInvites = 'failed';
        state.error = action.error.message;
      });
  },
});

export default invitationsSlice.reducer;
