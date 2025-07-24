import { Text, View, Image } from 'react-native';
import { useResponsiveHeight, useResponsiveWidth, useResponsiveFontSize } from 'react-native-responsive-dimensions';
import { sharedStylesHome, largeScreenStylesHome, smallScreenStylesHome } from '../styles/HomePageStyles';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { useSelector } from 'react-redux';
import { textStyles } from '../styles/textStyles';
import { useEffect } from 'react';
import CustomButton from '../components/CustomButton';
import stethoscopeImg from '../assets/images/stethoscope-clear.png';
import editImg from '../assets/images/edit-clear.png';
import checkImg from '../assets/images/check-clear.png';
import hourglass from '../assets/images/hourglass.png';
import * as Amplitude from '@amplitude/analytics-react-native';

export default function HomePage({ navigation }) {
  const user = useSelector((state) => state.user);

  const isLargeScreen = useWindowWidth();

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);

  const styles = {
    ...sharedStylesHome,
    ...(isLargeScreen ? largeScreenStylesHome : smallScreenStylesHome),
  };

  useEffect(() => {
    Amplitude.logEvent('OPEN_HOME_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_HOME_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  const handleSurveyNav = () => {
    navigation.navigate('SURVEYSELECT');
  };

  const layoutStyle = isLargeScreen ? styles.horizontalLayout : styles.verticalLayout;

  const handleApptsBtnClicked = () => {};

  const handleSurveyHistoryClicked = () => {
    navigation.navigate('SURVEYHISTORY');
  };

  const handleAssessmentHistoryClicked = () => {};

  return (
    <View style={styles.container}>
      <View
        style={{
          ...styles.centerContent,
          marginBottom: useResponsiveHeight(10),
        }}
      >
        <View
          style={{
            width: isLargeScreen ? useResponsiveWidth(100) : useResponsiveWidth(85),
            alignItems: 'left',
            paddingLeft: isLargeScreen ? 20 : 0,
          }}
        >
          <Text
            style={{
              ...styles.welcomeText,
              ...textStyles.regular,
              fontSize: useResponsiveFontSize(2.5),
            }}
          >
            {`Welcome back, ${user.getUserData?.firstName}`}
          </Text>
        </View>
        <View style={layoutStyle}>
          {HomePageBox(
            isLargeScreen,
            isLargeScreen ? useResponsiveHeight(38) : useResponsiveHeight(25),
            styles,
            '#3C6672',
            '#2B4851',
            'Surveys',
            editImg,
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={styles}
              buttonText={'Complete a Survey'}
              height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
              width={'85%'}
              marginTop={20}
              marginBottom={10}
              onPress={handleSurveyNav}
              buttonColor={'#F3F3F3'}
            />,
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={styles}
              buttonText={'View Survey History'}
              height={isLargeScreen ? useResponsiveHeight(9) : useResponsiveHeight(5)}
              width={'85%'}
              marginTop={10}
              marginBottom={20}
              onPress={handleSurveyHistoryClicked}
              buttonColor={'#F3F3F3'}
            />,
          )}
          {HomePageBox(
            isLargeScreen,
            isLargeScreen ? useResponsiveHeight(38) : useResponsiveHeight(21),
            styles,
            '#84A258',
            '#4D5640',
            'Assessments',
            checkImg,
            // Uncomment to enable this feature, then remove next View
            // ****
            // <CustomButton
            //   isLargeScreen={isLargeScreen}
            //   styles={styles}
            //   buttonText={'View Assessment History'}
            //   height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
            //   width={'85%'}
            //   marginTop={20}
            //   marginBottom={20}
            //   onPress={handleAssessmentHistoryClicked}
            //   buttonColor={'#F3F3F3'}
            // />,
            <View
              style={{
                backgroundColor: '#647847',
                width: '85%',
                height: buttonHeight,
                borderRadius: 10,
                flexDirection: 'row',
                justifyContent: 'center',
                alignItems: 'center',
              }}
            >
              <Image source={hourglass} style={{ height: buttonHeight - 20, resizeMode: 'contain' }}></Image>
              <Text
                style={{
                  color: 'white',
                  fontSize: useResponsiveFontSize(2.5),
                }}
              >
                Coming Soon
              </Text>
            </View>,
          )}
          {HomePageBox(
            isLargeScreen,
            isLargeScreen ? useResponsiveHeight(38) : useResponsiveHeight(21),
            styles,
            '#DEBE5B',
            '#6B5C2C',
            'Resources',
            stethoscopeImg,
            // Uncomment to enable this feature, then remove next View
            // ****
            // <CustomButton
            //   isLargeScreen={isLargeScreen}
            //   styles={styles}
            //   buttonText={'View & Book Appointments'}
            //   height={isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7)}
            //   width={'85%'}
            //   marginTop={20}
            //   marginBottom={20}
            //   onPress={handleApptsBtnClicked}
            //   buttonColor={'#F3F3F3'}
            // />,
            <View
              style={{
                backgroundColor: '#927C38',
                width: '85%',
                height: buttonHeight,
                borderRadius: 10,
                flexDirection: 'row',
                justifyContent: 'center',
                alignItems: 'center',
              }}
            >
              <Image source={hourglass} style={{ height: buttonHeight - 20, resizeMode: 'contain' }}></Image>
              <Text
                style={{
                  color: 'white',
                  fontSize: useResponsiveFontSize(2.5),
                }}
              >
                Coming Soon
              </Text>
            </View>,
          )}
        </View>
      </View>
    </View>
  );
}

function HomePageBox(
  isLargeScreen,
  height,
  styles,
  mainColour,
  secondaryColour,
  label,
  imgSource,
  btnComponent,
  btnComponentSecondary,
) {
  return (
    <View
      style={{
        ...styles.box,
        backgroundColor: mainColour,
        height: height,
        width: isLargeScreen ? useResponsiveWidth(31) : useResponsiveWidth(85),
      }}
    >
      <View
        style={{
          ...styles.labelContainer,
          backgroundColor: secondaryColour,
          height: isLargeScreen ? useResponsiveHeight(10) : useResponsiveHeight(7),
        }}
      >
        <Text style={{ ...styles.labelText, fontSize: useResponsiveFontSize(2.5) }}>{label}</Text>
      </View>
      <Image source={imgSource} style={styles.bgImage} />
      <View style={styles.btnsContainer}>
        {btnComponent}
        {btnComponentSecondary}
      </View>
    </View>
  );
}
