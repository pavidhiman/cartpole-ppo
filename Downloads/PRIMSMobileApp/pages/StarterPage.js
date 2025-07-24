import * as Amplitude from '@amplitude/analytics-react-native';
import { exchangeCodeAsync, useAuthRequest } from 'expo-auth-session';
import { useEffect, useState } from 'react';
import { Image, Text, TouchableOpacity, View } from 'react-native';
import { useResponsiveWidth } from 'react-native-responsive-dimensions';
import { useDispatch, useSelector } from 'react-redux';
import gif from '../assets/images/logo-loader-whitebg.gif';
import primsLanding from '../assets/images/PRIMS-landing.png';
import primsPattern from '../assets/images/PRIMSPattern.png';
import CustomButton from '../components/CustomButton';
import config from '../config.json';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { getUser } from '../slices/UserSlice';
import { getValidUser, updatePatientID } from '../slices/ValidUserSlice';
import { starterPageStyles } from '../styles/StarterPageStyles';
import {
  fetchTokenData,
  generateCodeChallenge,
  generateCodeVerifier,
  parseJwt,
  storeAccessToken,
  storeRefreshToken,
} from '../utils/Library';

export default function StarterPage({ navigation, route }) {
  const isLargeScreen = useWindowWidth();

  const [showWebView, setShowWebView] = useState(false);
  const [codeVerifier, setCodeVerifier] = useState(null);
  const [codeChallenge, setCodeChallenge] = useState(null);

  const validUser = useSelector((state) => state.validUser);
  const user = useSelector((state) => state.user);

  const [userAuthenticated, setUserAuthenticated] = useState(false);

  const expoTokenForStarter = route.params;

  const dispatch = useDispatch();
  useEffect(() => {
    if (validUser.status === 'succeeded') {
      dispatch(getUser({ userID: validUser.data.userID }));
    }
  }, [validUser]);

  useEffect(() => {
    /* TEMPORARY BYPASS for testing APIs
    console.log("Bypassing to HOME");
    navigation.reset({
      index: 0,
      routes: [{ name: 'HOME' }],
    });
  
    /*
    // ORIGINAL LOGIN FLOW — Uncomment when stable
    if (user.getUserStatus === 'succeeded') {
      if (user.getUserData.userSetupCompleted === 1 && user.getUserData.isEULA === 1) {
        Amplitude.logEvent('EXISTING_USER_LOGGED_IN');
        
        // UNCOMMENT THE FOLLOWING FOR PRODUCTION
        if (expoTokenForStarter?.data) {
          const pushTokenData = {
            UserID: user.getUserData.userID,
            PatientID: validUser.data.patientID,
            PushToken: expoTokenForStarter.data,
          };
          dispatch(addPushToken({ data: pushTokenData }));
        }
  
        navigation.reset({
          index: 0,
          routes: [{ name: 'HOME' }],
        });
      } else {
        dispatch(updateUserTypeID({ userID: validUser.data.userID }));
      }
    }
    */
  }, []);  

  useEffect(() => {
    if (user.updateUserTypeStatus === 'succeeded') {
      dispatch(updatePatientID(user.updateUserTypeData.patientID));
      if (validUser.data.patientID === user.updateUserTypeData.patientID) {
        // UNCOMMENT THE FOLLOWING FOR PRODUCTION
        // if (expoTokenForStarter?.data) {
        //   const pushTokenData = {
        //     UserID: user.getUserData.userID,
        //     PatientID: validUser.data.patientID,
        //     PushToken: expoTokenForStarter.data,
        //   };
        //   dispatch(addPushToken({ data: pushTokenData }));
        // }
        navigation.reset({
          index: 0,
          routes: [{ name: 'USERSETUP' }],
        });
      }
    }
  }, [user.updateUserTypeStatus, validUser]);

  useEffect(() => {
    (async () => {
      const generatedCodeVerifier = generateCodeVerifier();
      const generatedCodeChallenge = await generateCodeChallenge(generatedCodeVerifier);
      setCodeChallenge(generatedCodeChallenge);
      setCodeVerifier(generatedCodeVerifier);
    })();
  }, []);

  useEffect(() => {
    Amplitude.logEvent('OPEN_STARTER_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_STARTER_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  const handleAuthPressAndroid = () => {
    setShowWebView(true);
  };

  const discovery = {
    authorizationEndpoint: config.AUTH_ENDPOINT,
  };

  const [request, response, promptAsync] = useAuthRequest(
    {
      clientId: config.CLIENT_ID,
      scopes: [config.CUSTOM_SCOPE, 'offline_access'],
      redirectUri: 'primsapp://auth',
      extraParams: {
        code_challenge: codeChallenge,
        response_mode: 'query',
        prompt: 'login',
      },
      codeChallengeMethod: 'S256',
    },
    discovery,
  );

  const handleAuthPressGeneral = async () => {
    Amplitude.logEvent('USER_AUTHENTICATION_INITIATED');
    try {
      const authResult = await promptAsync();
      if (authResult.type === 'success') {
        setUserAuthenticated(true);
        // get the auth code
        const code = authResult.params.code;

        const discovery = {
          tokenEndpoint: config.TOKEN_ENDPOINT,
        };

        const exchangeConfig = {
          clientId: config.CLIENT_ID,
          code,
          redirectUri: 'primsapp://auth',
          extraParams: {
            code_verifier: codeVerifier,
          },
        };
        const fetchResult = await exchangeCodeAsync(exchangeConfig, discovery);
        if (fetchResult && fetchResult.accessToken && fetchResult.refreshToken) {
          console.log('access token: ', fetchResult.accessToken);
          console.log('refresh token: ', fetchResult.refreshToken);
          Amplitude.logEvent('USER_AUTHENTICATION_AND_CODE_EXCHANGE_SUCCEEDED');
          const storeAccessTokenResult = await storeAccessToken(fetchResult.accessToken);
          const storeRefreshTokenResult = await storeRefreshToken(fetchResult.refreshToken);
          if (storeAccessTokenResult && storeRefreshTokenResult) {
            // decode it so we can pass the email into ValidateUser
            const decodedToken = parseJwt(fetchResult.accessToken);
            dispatch(getValidUser({ email: decodedToken.emails[0] }));
          } else {
            setUserAuthenticated(false);
            console.log('Error retrieving or storing token');
          }
        } else {
          setUserAuthenticated(false);
          console.log(fetchResult);
        }
      } else {
        console.log(authResult);
      }
    } catch (error) {
      setUserAuthenticated(false);
      console.error('Error during authentication: ', error);
      Amplitude.logEvent('AUTHENTICATION_ERROR', {
        'error': error,
      });
    }
  };

  const handleWebViewNavigationStateChange = async (newNavState) => {
    const { url } = newNavState;
    if (url.includes('primsapp://auth') && !url.includes('authorize?')) {
      setShowWebView(false);
      const code = parseUrlForCode(url);
      // exchange authorization code for token data (access token & refresh token)
      const fetchResult = await fetchTokenData(code, codeVerifier);
      if (fetchResult && fetchResult.access_token && fetchResult.refresh_token) {
        const storeAccessTokenResult = await storeAccessToken(fetchResult.access_token);
        const storeRefreshTokenResult = await storeRefreshToken(fetchResult.refresh_token);
        if (storeAccessTokenResult && storeRefreshTokenResult) {
          dispatch(fetchAccessToken());
          // decode it so we can pass the email into ValidateUser
          const decodedToken = parseJwt(fetchResult.access_token);
          dispatch(getValidUser({ email: decodedToken.emails[0] }));
        } else {
          console.log('Error retrieving or storing token');
        }
      } else {
        console.log(fetchResult);
      }
    }
  };

  /*
  if (Platform.OS === 'android') {
    return showWebView ? (
      <WebView
        source={{
          uri:
            `${config.AUTH_ENDPOINT}?` +
            `client_id=${config.CLIENT_ID}` +
            `&scope=${config.CUSTOM_SCOPE} offline_access` +
            `&response_type=code` +
            `&redirect_uri=primsapp://auth` +
            `&code_challenge=${codeChallenge}` +
            `&code_challenge_method=S256` +
            `&response_mode=query`,
        }}
        onNavigationStateChange={handleWebViewNavigationStateChange}
      />
    ) : (
      <ContentComponent isLargeScreen={isLargeScreen} styles={starterPageStyles} onPress={handleAuthPressAndroid} navigation={navigation}/>
    );
  } else {
    */
  return userAuthenticated ? (
    <LoadingComponent /> // Replace this with the component you want to render when authenticated
  ) : (
    <ContentComponent isLargeScreen={isLargeScreen} styles={starterPageStyles} onPress={handleAuthPressGeneral} navigation={navigation} />
  );
  //}
}

const parseUrlForCode = (url) => {
  const regex = /code=([^&]*)/;
  const found = url.match(regex);
  return found && found[1];
};

function ContentComponent({ isLargeScreen, styles, onPress, navigation }) {
  return (
    <View style={styles.centerContainer}>
      <Image
        source={primsLanding}
        resizeMode="contain"
        style={{
          width: isLargeScreen ? 904 / 5 : 904 / 3.5,
          height: isLargeScreen ? 518 / 5 : 518 / 3.5,
          marginBottom: 60,
        }}
      />
      <Image source={primsPattern} resizeMode="contain" style={styles.patternImage} />

      {/* Regular Login Button */}
      <CustomButton
        isLargeScreen={false}
        styles={styles}
        buttonText={'Sign In'}
        height={65}
        width="60%"
        onPress={onPress}
        buttonColor={'#3C6672'}
      />

      {/* Trial Login Link */}
      <TouchableOpacity
        onPress={() => navigation.navigate('TRIALLOGIN')}
        style={{ marginTop: 30 }}
      >
        <Text style={{ color: '#3C6672', textAlign: 'center', fontSize: 16 }}>
          I have a Clinical-Trial ID
        </Text>
      </TouchableOpacity>
    </View>
  );
}

function LoadingComponent() {
  return (
    <View style={{ alignItems: 'center', justifyContent: 'center', flex: 1 }}>
      <Image source={gif} style={{ width: useResponsiveWidth(25), height: useResponsiveWidth(25) }}></Image>
    </View>
  );
}
