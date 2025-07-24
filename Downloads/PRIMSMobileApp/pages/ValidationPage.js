import * as Amplitude from '@amplitude/analytics-react-native';
import * as WebBrowser from 'expo-web-browser';
import { useEffect, useState } from 'react';
import { Image, View } from 'react-native';
import { useResponsiveWidth } from 'react-native-responsive-dimensions';
import { useDispatch, useSelector } from 'react-redux';
import gif from '../assets/images/logo-loader-whitebg.gif';
import { DEBUG_BYPASS_AUTH } from '../config.debug';
import { getUser } from '../slices/UserSlice';
import { getValidUser } from '../slices/ValidUserSlice';
import {
  deleteAccessToken,
  deleteRefreshToken,
  getAccessToken,
  getRefreshToken,
  isValidToken,
  parseJwt,
  refreshAccessToken,
  storeAccessToken,
  storeRefreshToken,
  tokenExpired,
} from '../utils/Library';

export default function ValidationPage({ navigation }) {
  const dispatch = useDispatch();
  const validUser = useSelector((state) => state.validUser);
  const user = useSelector((state) => state.user);

  const [expoToken, setExpoToken] = useState(null);

  useEffect(() => {
    if (validUser.status === 'succeeded') {
      dispatch(getUser({ userID: validUser.data.userID }));
    }

    if (validUser.status === 'failed') {
      console.error('Error occurred validating user', validUser.error);
      sendToStarterPage(navigation, expoToken);
    }
  }, [validUser.status]);

  useEffect(() => {
    if (user.getUserStatus === 'succeeded') {
      if (user.getUserData.userSetupCompleted === 1 && user.getUserData.isEULA === 1) {
        Amplitude.logEvent('EXISTING_USER_VALIDATED');
        // UNCOMMENT THE FOLLOWING FOR PRODUCTION
         if (expoToken?.data) {
           const pushTokenData = {
             UserID: user.getUserData.userID,
             PatientID: validUser.data.patientID,
             PushToken: expoToken.data,
           };
           dispatch(addPushToken({ data: pushTokenData }));
         }
        navigation.reset({
          index: 0,
          routes: [{ name: 'HOME' }],
        });
      } else {
        navigation.reset({
          index: 0,
          routes: [{ name: 'USERSETUP' }],
        });
      }
    }
    if (user.getUserStatus === 'failed') {
      console.error('Error occurred getting user', user.error);
      sendToStarterPage(navigation, expoToken);
    }
  }, [user.getUserStatus]);

  
  useEffect(() => {
    Amplitude.logEvent('VALIDATION_PAGE_OPENED');
    if (DEBUG_BYPASS_AUTH) { // DEBUGGING
      console.log('DEV-BYPASS: sending straight to STARTER screen');
      sendToStarterPage(navigation, null);
      return;             // stop the normal auth flow
    }
    const startUp = async () => {
      // Optional: Register for push notifications
      /*
      const pushToken = await registerForPushNotificationsAsync();
      Amplitude.logEvent('USER_PUSH_TOKEN_RESULT', { 'result': pushToken });
      setExpoToken(pushToken);
      */

      await deleteAccessToken();
      await deleteRefreshToken();

      const token = await getAccessToken();
      const refreshToken = await getRefreshToken();
      console.log('access token: ', token);
      console.log('refresh token: ', refreshToken);
  
      if (token && isValidToken(token)) {
        Amplitude.logEvent('TOKEN_EXISTS_AND_IS_VALID');
        if (tokenExpired(token)) {
          Amplitude.logEvent('TOKEN_IS_EXPIRED');
          const refreshResult = await refreshAccessToken(refreshToken);
          if (refreshResult?.access_token && refreshResult?.refresh_token) {
            await storeAccessToken(refreshResult.access_token);
            await storeRefreshToken(refreshResult.refresh_token);
            handleSetup(dispatch, refreshResult.access_token, expoToken, navigation);
          } else {
            Amplitude.logEvent('TOKEN_REFRESH_FAILED');
            await deleteAccessToken();
            await deleteRefreshToken();
            // await runLoginFlow(); // fallback to manual login
            sendToStarterPage(navigation, null);
          }
        } else {
          Amplitude.logEvent('TOKEN_VALID_AND_NOT_EXPIRED');
          handleSetup(dispatch, token, expoToken, navigation);
        }
      } else {
        Amplitude.logEvent('TOKEN_NOT_EXISTS_OR_NOT_VALID');
        // await runLoginFlow();
        sendToStarterPage(navigation, null);
      }
    };
  
    const runLoginFlow = async () => {
      // const authUrl = `https://pragmaclindevb2c.b2clogin.com/pragmaclindevb2c.onmicrosoft.com/b2c_1_signupsignindev/oauth2/v2.0/authorize?client_id=440fae66-9021-4615-a16f-7ac1f6fd4c6d&response_type=token&redirect_uri=primsapp://auth&scope=https://pragmaclindevb2c.onmicrosoft.com/primsapi/user.access offline_access`;
  
      
      const scopes = encodeURIComponent(
          'https://pragmaclindevb2c.onmicrosoft.com/primsapi/user.access'
        );
        const authUrl = `https://pragmaclindevb2c.b2clogin.com/pragmaclindevb2c.onmicrosoft.com/b2c_1_signupsignindev/oauth2/v2.0/authorize` +
          `?client_id=440fae66-9021-4615-a16f-7ac1f6fd4c6d` +
          `&response_type=token` +              
          `&redirect_uri=primsapp://auth` +
          `&scope=${scopes}`;
        
          console.log('opening Azure B2C');
          const result = await WebBrowser.openAuthSessionAsync(
            authUrl,
            'primsapp://auth'
          );
          console.log('browser result →', result);
  
      if (result.type === 'success' && result.url) {
        const params = new URLSearchParams(result.url.split("#")[1]);
        const accessToken = params.get("access_token");
        const refreshToken = params.get("refresh_token");
  
        if (accessToken) {
          await storeAccessToken(accessToken);
          if (refreshToken) await storeRefreshToken(refreshToken);
          handleSetup(dispatch, accessToken, expoToken, navigation);
        } else {
          Amplitude.logEvent('AUTH_TOKEN_MISSING_FROM_URL');
          sendToStarterPage(navigation, expoToken);
        }
      } else {
        Amplitude.logEvent('LOGIN_CANCELLED_OR_FAILED');
        sendToStarterPage(navigation, expoToken);
      }
    };
  
    startUp();
  }, []);
  

  /*
  // TEMP mock for testing
  useEffect(() => {
    dispatch(setValidUserMock({
      userID: 123,
      patientID: 456,
    }));
  
  
    console.log("Mocked validUser, navigating to HOME");
    navigation.reset({
      index: 0,
      routes: [{ name: 'HOME' }],
    });
  }, []);
  */

  return (
    <View style={{ alignItems: 'center', justifyContent: 'center', flex: 1 }}>
      <Image source={gif} style={{ width: useResponsiveWidth(25), height: useResponsiveWidth(25) }}></Image>
    </View>
  );
}

// handle setting up access and refresh tokens and retrieving valid user from db
function handleSetup(dispatch, token, expoToken, navigation) {
  try {
    // decode it so we can pass the email into ValidateUser
    const decodedToken = parseJwt(token);
    dispatch(getValidUser({ email: decodedToken.emails[0] }));
  } catch (error) {
    sendToStarterPage(navigation, expoToken);
  }
}

function sendToStarterPage(navigation, expoToken) {
  navigation.reset({
    index: 0,
    routes: [
      {
        name: 'STARTER',
        params: {
          expoTokenForStarter: expoToken,
        },
      },
    ],
  });
}
