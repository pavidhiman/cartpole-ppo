import { configureStore } from '@reduxjs/toolkit';
import { combineReducers } from 'redux';
import apptsReducer from './slices/AppointmentsSlice';
import assessmentsReducer from './slices/AssessmentSlice';
import authReducer from './slices/AuthSlice';
import conditionsReducer from './slices/ConditionsSlice';
import locationsReducer from './slices/LocationsSlice';
import settingsReducer from './slices/PatientSettingsSlice';
import pushTokenReducer from './slices/PushTokenSlice';
import surveysReducer from './slices/SurveySlice';
import userReducer from './slices/UserSlice';
import validUserReducer from './slices/ValidUserSlice';

const appReducer = combineReducers({
  validUser: validUserReducer,
  user: userReducer,
  locations: locationsReducer,
  conditions: conditionsReducer,
  settings: settingsReducer,
  auth: authReducer,
  surveys: surveysReducer,
  appts: apptsReducer,
  pushToken: pushTokenReducer,
  assessments: assessmentsReducer,
});

const initialState = appReducer({}, {});

const rootReducer = (state, action) => {
  if (action.type === 'USER_LOGOUT') {
    state = initialState;
  }

  return appReducer(state, action);
};

const store = configureStore({
  reducer: rootReducer,
});

export default store;
