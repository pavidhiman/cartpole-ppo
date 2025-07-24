import { Text, Pressable, Image, View } from 'react-native';
import { useResponsiveFontSize } from 'react-native-responsive-dimensions';
import { useResponsiveHeight, useResponsiveWidth } from 'react-native-responsive-dimensions';
import React, { useState, useEffect } from 'react';
import { textStyles } from '../../styles/textStyles';
import CustomButton from '../CustomButton';
import { popupStyles } from '../Popup/PopupStyles.js';

export default function Popup({
  text,
  header,
  buttonAmount,
  firstButtonText,
  secondButtonText,
  coloredButton,
  theme,
  isLargeScreen,
  leftFunction,
  rightFunction,
  singleButtonWidth,
  marginTop,
  marginBottom,
}) {
  const [colorCode, setColorCode] = useState('');

  const presetStyles = {
    ...popupStyles,
    // ...(isLargeScreen ? largeScreenStylesHome : smallScreenStylesHome),
  };

  const colors = (theme) => {
    switch (theme) {
      case 'general':
        setColorCode('#3C6672');
        break;
      case 'appointment':
        setColorCode('#6B5C2C');
        break;
      case 'survey':
        setColorCode('#3C6672');
        break;
      case 'assessment':
        setColorCode('#4D5640');
        break;
    }
  };

  useEffect(() => {
    colors(theme);
  }, []);

  return (
    <View style={presetStyles.tint}>
      <View
        style={{
          ...popupStyles.popupFrame,
          minHeight: useResponsiveHeight(30),
          width: useResponsiveWidth(90),
          marginTop: marginTop,
          marginBottom: marginBottom,
        }}
      >
        {header !== undefined && (
          <Text style={{ ...popupStyles.popupHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            {header}
          </Text>
        )}

        <Text style={{ ...popupStyles.popupText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.5) }}>
          {text}
        </Text>
        <View style={popupStyles.buttonWrapper}>
          {buttonAmount >= 1 && (
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={presetStyles}
              buttonText={firstButtonText}
              height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
              width={singleButtonWidth ? useResponsiveWidth(singleButtonWidth) : useResponsiveWidth(30)}
              buttonColor={coloredButton == 'left' ? colorCode : undefined}
              onPress={leftFunction}
            ></CustomButton>
          )}
          {buttonAmount == 2 && (
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={presetStyles}
              buttonText={secondButtonText}
              height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
              width={useResponsiveWidth(30)}
              buttonColor={coloredButton == 'right' ? colorCode : undefined}
              onPress={rightFunction}
            ></CustomButton>
          )}
        </View>
      </View>
    </View>
  );
}
