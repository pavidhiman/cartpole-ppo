import * as Amplitude from '@amplitude/analytics-react-native';
import axios from 'axios';
import Constants from 'expo-constants';
import * as Crypto from 'expo-crypto';
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';
import * as SecureStore from "expo-secure-store";
import { Base64 } from 'js-base64';
import jwtDecode from 'jwt-decode';
import { Platform } from 'react-native';
import config from '../config.json';

export function tokenExpired(token) {
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;

    return decoded.exp < currentTime;
  } catch (e) {
    console.error('Invalid token:', e);
    return true;
  }
}

export function isValidToken(token) {
  try {
    jwtDecode(token);
    return true;
  } catch (e) {
    console.error('Invalid token:', e);
    Amplitude.logEvent('INVALID_ACCESS_TOKEN_ERROR', {
      'error': error,
    });
    return false;
  }
}

export function parseJwt(token) {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');

    // Decode using the js-base64 library
    const payload = JSON.parse(Base64.decode(base64));

    return payload;
  } catch (error) {
    console.error('Failed to decode JWT: ', error);
    return null;
  }
}

export async function storeAccessToken(accessToken) {
  try {
    // change for prod vs dev
    await SecureStore.setItemAsync('access_token_production', accessToken);
    return true;
  } catch (e) {
    console.error('Error saving access token to SecureStore', e);
    Amplitude.logEvent('STORING_ACCESS_TOKEN_ERROR', {
      'error': error,
    });
    return false;
  }
}


// comment out when not using web
export async function getAccessToken() {
  try {
    // change for prod vs dev
    return await SecureStore.getItemAsync('access_token_production');
  } catch (e) {
    console.error('Error fetching access token from SecureStore', e);
    Amplitude.logEvent('GET_ACCESS_TOKEN_ERROR', {
      'error': error,
    });
  }
}

export async function storeRefreshToken(refreshToken) {
  try {
    // change for prod vs dev
    await SecureStore.setItemAsync('refresh_token_production', refreshToken);
    return true;
  } catch (e) {
    console.error('Error saving refresh token to SecureStore', e);
    Amplitude.logEvent('STORING_REFRESH_TOKEN_ERROR', {
      'error': error,
    });
    return false;
  }
}

export async function getRefreshToken() {
  try {
    // change for prod vs dev
    return await SecureStore.getItemAsync('refresh_token_production');
  } catch (e) {
    console.error('Error fetching refresh token from SecureStore', e);
    Amplitude.logEvent('GET_REFRESH_TOKEN_ERROR', {
      'error': error,
    });
  }
}

export async function deleteAccessToken() {
  try {
    // change for prod vs dev
    await SecureStore.deleteItemAsync('access_token_production');
  } catch (error) {
    console.log('Could not delete access token', error);
  }
}

export async function deleteRefreshToken() {
  try {
    // change for prod vs dev
    await SecureStore.deleteItemAsync('refresh_token_production');
  } catch (error) {
    console.log('Could not delete refresh token', error);
  }
}

export async function fetchTokenData(code, codeVerifier) {
  const tokenEndpoint = config.TOKEN_ENDPOINT;
  const client_id = config.CLIENT_ID;
  const redirect_uri = 'primsapp://auth';
  const grant_type = 'authorization_code';
  const code_challenge_method = 'S256';
  const scope = 
    `${config.CUSTOM_SCOPE} offline_access`;   

  const data = new URLSearchParams({
    client_id,
    redirect_uri,
    grant_type,
    code,
    code_verifier: codeVerifier,
    code_challenge_method,
    scope,
  });

  try {
    const response = await axios({
      method: 'post',
      url: tokenEndpoint,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      data: data.toString(),
    });

    return response.data;
  } catch (error) {
    console.error('Error fetching token data:', error);
    if (error.response) {
      console.error(error.response.data);
    }
    return null;
  }
}

export function generateRandomString(length) {
  const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let result = '';
  while (length > 0) {
    const randomIndex = Math.floor(Math.random() * charset.length);
    result += charset[randomIndex];
    length--;
  }
  return result;
}

export function generateCodeVerifier(minLength = 43, maxLength = 128) {
  return generateRandomString(Math.floor(Math.random() * (maxLength - minLength + 1)) + minLength);
}

export async function generateCodeChallenge(verifier) {
  const hashed = await Crypto.digestStringAsync(Crypto.CryptoDigestAlgorithm.SHA256, verifier, {
    encoding: Crypto.CryptoEncoding.BASE64,
  });

  return hashed.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export async function refreshAccessToken(refreshToken) {
  const tokenEndpoint = config.TOKEN_ENDPOINT;
  const client_id = config.CLIENT_ID;
  const scope = `${config.CUSTOM_SCOPE} offline_access`;
  const grant_type = 'refresh_token';

  const data = new URLSearchParams({
    client_id,
    scope,
    grant_type,
    refresh_token: refreshToken,
  });

  try {
    const response = await axios({
      method: 'post',
      url: tokenEndpoint,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      data: data.toString(),
    });

    return response.data;
  } catch (error) {
    console.error('Error refreshing token:', error);
    Amplitude.logEvent('REFRESH_TOKEN_ERROR', {
      'error': error,
    });
    return null;
  }
}

export async function registerForPushNotificationsAsync() {
  let token = null;

  try {
    if (Platform.OS === 'android') {
      await Notifications.setNotificationChannelAsync('default', {
        name: 'default',
        importance: Notifications.AndroidImportance.MAX,
        vibrationPattern: [0, 250, 250, 250],
        lightColor: '#FF231F7C',
      });
    }

    if (Device.isDevice) {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;
      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }
      if (finalStatus !== 'granted') {
        Amplitude.logEvent('FAILED_TO_GET_PUSH_TOKEN');
        console.log('Failed to get push token for push notification');
        return null;
      }
      token = await Notifications.getExpoPushTokenAsync({
        projectId: Constants.expoConfig.extra.eas.projectId,
      });
    } else {
      console.log('Must use physical device for Push Notifications');
    }
  } catch (error) {
    Amplitude.logEvent('PUSH_NOTIFICATIONS_ERROR', {
      'error': error,
    });
    return null;
  }

  return token;
}

export async function handleTokenRefresh() {
  const refreshToken = await getRefreshToken();
  const refreshResult = await refreshAccessToken(refreshToken);
  if (refreshResult && refreshResult.access_token && refreshResult.refresh_token) {
    const storeAccessTokenResult = await storeAccessToken(refreshResult.access_token);
    const storeRefreshTokenResult = await storeRefreshToken(refreshResult.refresh_token);
    if (storeAccessTokenResult && storeRefreshTokenResult) {
      return refreshResult.access_token;
    }
  }
  return null;
}