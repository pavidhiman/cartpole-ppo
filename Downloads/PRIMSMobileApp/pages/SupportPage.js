import { Text, View } from 'react-native';
import React, { useState } from 'react';
import { useResponsiveWidth, useResponsiveHeight, useResponsiveFontSize } from 'react-native-responsive-dimensions';
import { supportPageStyles } from '../styles/SupportPageStyles';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { useDispatch } from 'react-redux';
import { useEffect } from 'react';
import CustomButton from '../components/CustomButton';
import Popup from '../components/Popup/Popup';
import { deleteAccessToken, deleteRefreshToken } from '../utils/Library';
import { textStyles } from '../styles/textStyles';
import { CommonActions } from '@react-navigation/native';
import * as WebBrowser from 'expo-web-browser';
import logoutImage from '../assets/images/logout-icon.png';
import aboutImage from '../assets/images/about-icon.png';
import privacyImage from '../assets/images/privacy-icon.png';
import HorizontalRule from '../components/HorizontalRule';
import config from '../config.json';
import * as Amplitude from '@amplitude/analytics-react-native';

export default function SupportPage({ navigation }) {
  const dispatch = useDispatch();

  const isLargeScreen = useWindowWidth();

  const styles = {
    ...supportPageStyles,
  };

  const [logoutPopup, setLogoutPopup] = useState(false);

  const handleLogoutPopup = () => {
    setLogoutPopup(!logoutPopup);
  };

  useEffect(() => {
    Amplitude.logEvent('OPEN_SUPPORT_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_SUPPORT_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  const handleLogOutPress = async () => {
    const logoutUrl = config.LOGOUT_URL;
    const result = await WebBrowser.openAuthSessionAsync(logoutUrl, 'primsapp://auth');
    if (result.type === 'success') {
      await deleteAccessToken();
      await deleteRefreshToken();
      setLogoutPopup(!logoutPopup);
      dispatch({ type: 'USER_LOGOUT' });
      navigation.reset({
        index: 0,
        routes: [{ name: 'VALIDATE' }],
      });
    }
  };

  const handlePrivacyNav = () => {
    navigation.navigate('PRIVACY');
  };

  const handleAboutNav = () => {
    navigation.navigate('ABOUT');
  };

  return (
    <View style={styles.testContainer}>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            ACCOUNT/SUPPORT
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Take actions and access important information about your account.
          </Text>
        </View>
      </View>
      {logoutPopup && (
        <Popup
          text={'Are you sure you want to log out of your account?'}
          header={'Log Out'}
          buttonAmount={2}
          firstButtonText={'No'}
          secondButtonText={'Yes'}
          coloredButton={'right'}
          theme={'general'}
          leftFunction={handleLogoutPopup}
          rightFunction={handleLogOutPress}
          isLargeScreen={isLargeScreen}
        ></Popup>
      )}

      <View style={styles.mainContent}>
        <View style={styles.buttonWrapper}>
          <CustomButton
            isLargeScreen={isLargeScreen}
            styles={styles}
            buttonText={'Privacy Policy'}
            height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
            width={'85%'}
            marginTop={10}
            marginBottom={10}
            icon={privacyImage}
            onPress={handlePrivacyNav}
          />
          <CustomButton
            isLargeScreen={isLargeScreen}
            styles={styles}
            buttonText={'About'}
            height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
            width={'85%'}
            marginTop={10}
            marginBottom={10}
            icon={aboutImage}
            onPress={handleAboutNav}
          />

          <View style={{ ...styles.logoutWrapper, paddingTop: 50 }}>
            <HorizontalRule color="#E8E8E8" />
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={styles}
              buttonText={'Log Out'}
              height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
              marginTop={20}
              marginBottom={10}
              icon={logoutImage}
              onPress={handleLogoutPopup}
            />
          </View>
        </View>
      </View>
    </View>
  );
}
