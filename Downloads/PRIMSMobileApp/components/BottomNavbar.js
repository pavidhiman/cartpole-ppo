import { View, Image, Pressable, StyleSheet } from 'react-native';
import React, { useState } from 'react';
import { useResponsiveHeight, useResponsiveWidth } from 'react-native-responsive-dimensions';
import homeImg from '../assets/images/mobile-home-icon.png';
import profileImg from '../assets/images/mobile-profile-icon.png';
import settingsImg from '../assets/images/mobile-settings-icon.png';
import Popup from '../components/Popup/Popup';

import { useSelector } from 'react-redux';

export default function BottomNavbar({ onChangeScreen, currentScreen, changedSettings }) {
  const [surveyPopup, setSurveyPopup] = useState(false);
  const [settingsChangedPopup, setSettingsChangedPopup] = useState(false);
  const [intendedRoute, setIntendedRoute] = useState('');

  const surveys = useSelector((state) => state.surveys);

  const handleNavigateSort = (route) => {
    setIntendedRoute(route);
    //console.log(currentScreen);
    if (currentScreen === 'SURVEY') {
      setSurveyPopup(true);
      //console.log('match');
      //console.log(surveyPopup);
    } else if (currentScreen === 'SETTINGS' && changedSettings) {
      setSettingsChangedPopup(true);
    } else {
      onChangeScreen(route);
    }
  };

  return (
    <>
      {surveyPopup && (
        <Popup
          text={`Are you sure you want to quit your ${surveys.getSurveysData[0].surveyDesc}? Your progress won’t be saved.`}
          header={'Quit Survey'}
          buttonAmount={2}
          firstButtonText={'No'}
          secondButtonText={'Yes'}
          coloredButton={'right'}
          theme={'survey'}
          leftFunction={() => {
            setSurveyPopup(!surveyPopup);
          }}
          rightFunction={() => {
            onChangeScreen(intendedRoute);
            setSurveyPopup(false);
          }}
        ></Popup>
      )}
      {settingsChangedPopup && (
        <Popup
          text={
            'You have unsaved changes in your settings. Are you sure you want to exit without saving the changes first?'
          }
          header={'Unsaved Changes'}
          buttonAmount={2}
          firstButtonText={'No'}
          secondButtonText={'Yes'}
          coloredButton={'right'}
          theme={'survey'}
          leftFunction={() => {
            setSettingsChangedPopup(!settingsChangedPopup);
          }}
          rightFunction={() => {
            onChangeScreen(intendedRoute);
            setSettingsChangedPopup(false);
          }}
        ></Popup>
      )}
      <View
        style={{
          ...navbarStyles.bottomBar,
          height: useResponsiveHeight(10),
          zIndex: 100,
          justifyContent: 'space-evenly',
          paddingLeft: 20,
          paddingRight: 20,
        }}
      >
        <Pressable
          onPress={() => {
            handleNavigateSort('SETTINGS');
          }}
          style={{
            ...navbarStyles.barButton,
            backgroundColor: 'transparent',
          }}
        >
          <Image
            source={settingsImg}
            style={{
              ...navbarStyles.buttonImage,
              width: useResponsiveWidth(10),
              height: useResponsiveHeight(10),
            }}
          ></Image>
        </Pressable>
        <Pressable
          onPress={() => {
            handleNavigateSort('HOME');
          }}
          style={{
            height: useResponsiveWidth(14),
            width: useResponsiveWidth(14),
            borderRadius: 99,
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: '#D9D9D9',
            marginBottom: useResponsiveHeight(6),
          }}
        >
          <Image
            source={homeImg}
            style={{
              ...navbarStyles.buttonImage,
              width: useResponsiveWidth(12),
              height: useResponsiveHeight(12),
            }}
          ></Image>
        </Pressable>
        <Pressable
          onPress={() => {
            handleNavigateSort('SUPPORT');
          }}
          style={{
            ...navbarStyles.barButton,
            backgroundColor: 'transparent',
          }}
        >
          <Image
            source={profileImg}
            style={{
              ...navbarStyles.buttonImage,
              width: useResponsiveWidth(9.5),
              height: useResponsiveHeight(9.5),
            }}
          ></Image>
        </Pressable>
      </View>
    </>
  );
}

const navbarStyles = StyleSheet.create({
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: '#F3F3F3',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  barButton: {
    marginHorizontal: 10,
    paddingVertical: 5,
    paddingHorizontal: 15,
    borderRadius: 10,
    backgroundColor: 'white',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonImage: {
    resizeMode: 'contain',
  },
});
