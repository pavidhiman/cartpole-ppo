import { useRef, useState, useEffect } from 'react';
import { View, Text, FlatList, ScrollView, Pressable, Image, Platform, Switch } from 'react-native';
import {
  useResponsiveWidth,
  useResponsiveHeight,
  useResponsiveFontSize,
  useResponsiveScreenFontSize,
} from 'react-native-responsive-dimensions';
import { userSetupPageStyles } from '../styles/UserSetupPageStyles';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { useSelector, useDispatch } from 'react-redux';
import { getLocations, setSelectedLocationRedux } from '../slices/LocationsSlice';
import { getConditions, setSelectedConditionRedux } from '../slices/ConditionsSlice';
import { updateEula, savePatientSettings } from '../slices/PatientSettingsSlice';
import { getUser, setUserSetupCompleted, updateUserAccountStatus } from '../slices/UserSlice';
import { textStyles } from '../styles/textStyles';
import DropDownPicker from 'react-native-dropdown-picker';
import CustomButton from '../components/CustomButton';
import CheckBox from 'expo-checkbox';
import primsLogo from '../assets/images/PRIMS-Logo.png';
import * as Amplitude from '@amplitude/analytics-react-native';
import { TouchableOpacity } from 'react-native';

export default function UserSetupPage({ navigation }) {
  const styles = userSetupPageStyles;
  const isLargeScreen = useWindowWidth();
  const [currentScreen, setCurrentScreen] = useState('agreement');
  const [hasPD, setHasPD] = useState(false);

  const locations = useSelector((state) => state.locations);
  const conditions = useSelector((state) => state.conditions);
  const validUser = useSelector((state) => state.validUser);
  const user = useSelector((state) => state.user);
  const settings = useSelector((state) => state.settings);

  const dispatch = useDispatch();
  useEffect(() => {
    const callAPIs = async () => {
      dispatch(getLocations());
      dispatch(getConditions());
      dispatch(getUser({ userID: validUser.data.userID }));
    };
    callAPIs();
  }, []);

  useEffect(() => {
    if (settings.savePatientSettingsStatus === 'succeeded') {
      dispatch(setUserSetupCompleted({ userID: user.getUserData.userID, setupCompleted: 1 }));
    }
  }, [settings.savePatientSettingsStatus]);

  useEffect(() => {
    if (user.setUserSetupCompletedStatus === 'succeeded') {
      navigation.reset({
        index: 0,
        routes: [{ name: 'HOME' }],
      });
    }
  }, [user.setUserSetupCompletedStatus]);

  useEffect(() => {
    Amplitude.logEvent('OPEN_USER_SETUP_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_USER_SETUP_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  // DEBUGGING
  useEffect(() => {
    if (validUser && validUser.data) {
      console.log('validUser.data in UserSetupPage:', validUser.data);
    }
  }, [validUser]);

  if (currentScreen === 'agreement') {
    if (user.getUserStatus === 'succeeded') {
      return <AgreementSection styles={styles} isLargeScreen={isLargeScreen} setState={setCurrentScreen} user={user} />;
    }
  } else if (currentScreen === 'accountStatus') {
    return (
      <AccountStatusSection
        styles={styles}
        isLargeScreen={isLargeScreen}
        user={user}
        setState={setCurrentScreen}
        setHasPD={setHasPD}
      />
    );
  } else if (currentScreen === 'patientPreference') {
    if (locations.getLocationsStatus === 'succeeded' && conditions.getConditionsStatus === 'succeeded') {
      return (
        <PatientPreferenceSection
          styles={styles}
          isLargeScreen={isLargeScreen}
          setState={setCurrentScreen}
          locationsArr={locations}
          conditionsArr={conditions}
        />
      );
    }
  } else {
    return (
      <NotificationsSection
        styles={styles}
        isLargeScreen={isLargeScreen}
        user={user}
        validUser={validUser}
        locations={locations}
        conditions={conditions}
        hasPD={hasPD}
      />
    );
  }
}

function NotificationsSection({ styles, isLargeScreen, user, validUser, locations, conditions, hasPD }) {
  const [notificationsToggled, setNotificationsToggled] = useState(true);
  const [textNotificationsToggled, setTextNotificationsToggled] = useState(true);
  const [emailNotificationsToggled, setEmailNotificationsToggled] = useState(true);
  const [pushNotificationsToggled, setPushNotificationsToggled] = useState(true);

  const conditionsArray = conditions.getConditionsData.map((condition) => {
    return {
      conditionID: condition.conditionID,
      condition: condition.condition,
    };
  });

  const conditionToID = conditions.getConditionsData.reduce((acc, condition) => {
    acc[condition.condition] = condition.conditionID;
    return acc;
  }, {});

  const locationToID = locations.getLocationsData.reduce((acc, location) => {
    acc[location.locationName] = location.locationID;
    return acc;
  }, {});

  const dispatch = useDispatch();
  const handleCompleteButtonClicked = () => {
    let patientConditionsArray = [];
    //for (let condition of conditions.selectedCondition) {
    //  patientConditionsArray.push({ patientID: validUser.data.patientID, conditionID: conditionToID[condition] });
    //}
    if (hasPD) {
      patientConditionsArray.push({ patientID: validUser.data.patientID, conditionID: 100 });
    } else {
      patientConditionsArray.push({ patientID: validUser.data.patientID, conditionID: 103 });
    }
    const data = {
      patientId: validUser.data.patientID,
      patientConditions: patientConditionsArray,
      conditions: conditionsArray,
      patient: {
        patientID: validUser.data.patientID,
        userID: user.getUserData.userID,
        preferredLocationID: 100,
        preferredLocation: 'Genesis Centre',
        firstName: user.getUserData.firstName,
        lastName: user.getUserData.lastName,
        email: user.getUserData.email,
        dob: user.getUserData.dob,
        conditions: '',
        lastAssessmentDate: user.getUserData.dob,
      },
      notificationSettings: {
        userID: user.getUserData.userID,
        receiveNotifications: notificationsToggled ? 1 : 0,
        emailNotices: emailNotificationsToggled ? 1 : 0,
        textNotices: textNotificationsToggled ? 1 : 0,
        appNotices: pushNotificationsToggled ? 1 : 0,
      },
    };
    dispatch(savePatientSettings({ data: data }));
    const surveyDurationMS = new Date().getTime() - new Date(user.userSetupStartTime).getTime();
    const surveyDurationSeconds = surveyDurationMS / 1000;
    Amplitude.logEvent('NEW_USER_SETUP_FINISHED', {
      'setup_duration': surveyDurationSeconds,
    });
  };

  const handleTextNotifsToggled = () => {
    setTextNotificationsToggled(!textNotificationsToggled);
  };

  const handleEmailNotifsToggled = () => {
    setEmailNotificationsToggled(!emailNotificationsToggled);
  };

  const handlePushNotifsToggled = () => {
    setPushNotificationsToggled(!pushNotificationsToggled);
  };

  const handleNotifsToggled = () => {
    setEmailNotificationsToggled(!notificationsToggled);
    setTextNotificationsToggled(!notificationsToggled);
    setPushNotificationsToggled(!notificationsToggled);
    setNotificationsToggled(!notificationsToggled);
  };

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);
  const buttonWidth = useResponsiveWidth(65);

  return (
    <View>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            PROFILE SETUP
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Select your preferred notification settings. Notifications will remind you of upcoming assessments and
            questionnaires.
          </Text>
        </View>
      </View>
      <View
        style={{
          alignItems: 'center',
          height: useResponsiveHeight(75),
          marginTop: 50,
        }}
      >
        <Text style={{ fontSize: useResponsiveScreenFontSize(3), ...textStyles.regular, marginBottom: 20 }}>
          NOTIFICATIONS
        </Text>
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
          <Text style={{ marginRight: 20, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Toggle notifications off/on
          </Text>
          <Switch
            value={notificationsToggled}
            onValueChange={handleNotifsToggled}
            trackColor={{ false: '#D9D9D9', true: '#3C6672' }}
            thumbColor="white"
          />
        </View>
        <View
          style={{
            borderWidth: 1,
            borderRadius: 10,
            borderColor: '#D9D9D9',
            paddingTop: 20,
            paddingBottom: 20,
            alignItems: 'center',
            justifyContent: 'center',
            width: useResponsiveWidth(80),
            height: useResponsiveHeight(30),
          }}
        >
          <Pressable onPress={handleTextNotifsToggled} style={{ marginBottom: 20 }}>
            <View style={{ ...styles.checkboxContainer, paddingTop: 15, paddingBottom: 15 }}>
              <Text style={{ fontSize: useResponsiveFontSize(2.15), ...textStyles.regular, flex: 1 }}>
                Text Notifications
              </Text>
              <CheckBox value={textNotificationsToggled} onValueChange={handleTextNotifsToggled} />
            </View>
          </Pressable>
          <Pressable onPress={handleEmailNotifsToggled} style={{ marginBottom: 20 }}>
            <View style={{ ...styles.checkboxContainer, paddingTop: 15, paddingBottom: 15 }}>
              <Text style={{ fontSize: useResponsiveFontSize(2.15), ...textStyles.regular, flex: 1 }}>
                Email Notifications
              </Text>
              <CheckBox value={emailNotificationsToggled} onValueChange={handleEmailNotifsToggled} />
            </View>
          </Pressable>
          <Pressable onPress={handlePushNotifsToggled}>
            <View style={{ ...styles.checkboxContainer, paddingTop: 15, paddingBottom: 15 }}>
              <Text style={{ fontSize: useResponsiveFontSize(2.15), ...textStyles.regular, flex: 1 }}>
                App Push Notifications
              </Text>
              <CheckBox value={pushNotificationsToggled} onValueChange={handlePushNotifsToggled} />
            </View>
          </Pressable>
        </View>
        <View style={{ alignItems: 'center' }}>
          <CustomButton
            isLargeScreen={false}
            styles={styles}
            buttonText={'Complete'}
            height={buttonHeight}
            width={buttonWidth}
            onPress={handleCompleteButtonClicked}
            buttonColor={'#3C6672'}
            marginTop={45}
          />
        </View>
      </View>
    </View>
  );
}

function AccountTypeSection({ styles }) {
  const [patientSelected, setPatientSelected] = useState(false);
  const [caregiverSelected, setCaregiverSelected] = useState(false);
  const [familyMemberSelected, setFamilyMemberSelected] = useState(false);
  const [selectedType, setSelectedType] = useState(null);

  const handlePatientSelected = () => {
    setPatientSelected(!patientSelected);
    setCaregiverSelected(false);
    setFamilyMemberSelected(false);
    setSelectedType(1);
  };

  const handleCareGiverSelected = () => {
    setCaregiverSelected(!caregiverSelected);
    setPatientSelected(false);
    setFamilyMemberSelected(false);
    setSelectedType(5);
  };

  const handleFamilyMemberSelected = () => {
    setFamilyMemberSelected(!familyMemberSelected);
    setPatientSelected(false);
    setCaregiverSelected(false);
    setSelectedType(4);
  };

  return (
    <View>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            PROFILE SETUP
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Choose an account type.
          </Text>
        </View>
      </View>
      <View style={{ ...styles.contentContainer, height: useResponsiveHeight(55) }}>
        <Text style={{ ...textStyles.regular, fontSize: useResponsiveFontSize(2.15), marginBottom: 15 }}>
          Select your account type
        </Text>
        <AccountTypeSelectionBox
          styles={styles}
          buttonText="Patient"
          descriptionText="Complete your own surveys and book your own appointments. Only you have access to your account."
          marginBottom={15}
          selected={patientSelected}
          onPress={handlePatientSelected}
        />
        <AccountTypeSelectionBox
          styles={styles}
          buttonText="Caregiver"
          descriptionText="Link your account to an existing patient account. You will be able to complete surveys and book appointments on the patient's behalf."
          marginBottom={15}
          selected={caregiverSelected}
          onPress={handleCareGiverSelected}
        />
        <AccountTypeSelectionBox
          styles={styles}
          buttonText="Family Member"
          descriptionText="Link your account to an existing patient account. You will be able to complete surveys and book appointments on the patient's behalf."
          selected={familyMemberSelected}
          onPress={handleFamilyMemberSelected}
        />
      </View>
    </View>
  );
}

function AccountStatusSection({ styles, isLargeScreen, user, setState, setHasPD }) {
  const statusData = [
    { label: 'I am testing the app', value: 'Tester' },
    { label: 'I have been diagnosed with Parkinson’s disease', value: 'IsPWP' },
    {
      label: 'I have been diagnosed with a neurological condition other than Parkinson’s disease',
      value: 'HasCondition',
    },
    { label: 'I suspect I may have a neurological condition', value: 'SuspectedCondition' },
  ];

  const [selectedStatusOpen, setSelectedStatusOpen] = useState(false);
  const [selectedStatus, setSelectedStatus] = useState(null);
  const [statuses, setStatuses] = useState(statusData);

  const dispatch = useDispatch();
  const handleNextButtonClicked = () => {
    dispatch(
      updateUserAccountStatus({
        userID: user.getUserData.userID,
        isPWP: selectedStatus === 'IsPWP' ? 1 : 0,
        hasCondition: selectedStatus === 'HasCondition' ? 1 : 0,
        suspectedCondition: selectedStatus === 'SuspectedCondition' ? 1 : 0,
        isTester: selectedStatus === 'Tester' ? 1 : 0,
      }),
    );
    setHasPD(selectedStatus === 'IsPWP' ? true : false);
    setState('notifications');
  };

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);
  const buttonWidth = useResponsiveWidth(65);

  return (
    <View>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            PROFILE SETUP
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Choose an account status.
          </Text>
        </View>
      </View>
      <View
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          height: useResponsiveHeight(50),
        }}
      >
        <View style={{ width: useResponsiveWidth(85), alignItems: 'center' }}>
          <Text style={{ ...styles.pickerLabel, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            SELECT YOUR ACCOUNT STATUS
          </Text>
          <DropDownPicker
            open={selectedStatusOpen}
            value={selectedStatus}
            items={statuses}
            setOpen={setSelectedStatusOpen}
            setValue={setSelectedStatus}
            setItems={setStatuses}
            placeholder={''}
            style={{
              zIndex: 1,
              borderColor: '#3C6672',
              borderRadius: 15,
            }}
            dropDownContainerStyle={{
              maxHeight: 170,
            }}
          />
        </View>
      </View>
      {selectedStatus !== null ? (
        <View style={{ alignItems: 'center', zIndex: -2, elevation: -2 }}>
          <CustomButton
            isLargeScreen={false}
            styles={styles}
            buttonText={'Next'}
            height={buttonHeight}
            width={buttonWidth}
            onPress={handleNextButtonClicked}
            buttonColor={'#3C6672'}
          />
        </View>
      ) : null}
    </View>
  );
}

function PatientPreferenceSection({ styles, isLargeScreen, setState, locationsArr, conditionsArr }) {
  // map the data to labels and values for the use state in AccountTypeSection
  const locationsData = locationsArr.getLocationsData.map((location) => ({
    label: location.locationName,
    value: location.locationName,
  }));
  const conditionsData = conditionsArr.getConditionsData.map((condition) => ({
    label: condition.condition,
    value: condition.condition,
  }));

  const [selectedConditionOpen, setSelectedConditionOpen] = useState(false);
  const [selectedCondition, setSelectedCondition] = useState(null);
  const [conditions, setConditions] = useState(conditionsData);

  const [selectedLocationOpen, setSelectedLocationOpen] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [locations, setLocations] = useState(locationsData);

  const dispatch = useDispatch();
  const handleNextButtonClicked = () => {
    dispatch(setSelectedConditionRedux(selectedCondition));
    dispatch(setSelectedLocationRedux(selectedLocation));
    setState('notifications');
  };

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);
  const buttonWidth = useResponsiveWidth(65);

  return (
    <View>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            PROFILE SETUP
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Choose an account type.
          </Text>
        </View>
      </View>
      <View style={{ ...styles.contentContainer, height: useResponsiveHeight(55) }}>
        <Text style={{ ...styles.pickerLabel, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
          SELECT YOUR CONDITION(S)
        </Text>
        <View
          style={{
            marginBottom: selectedConditionOpen ? 150 : 50,
            width: isLargeScreen ? useResponsiveWidth(35) : useResponsiveWidth(85),
          }}
        >
          <DropDownPicker
            open={selectedConditionOpen}
            value={selectedCondition}
            items={conditions}
            setOpen={setSelectedConditionOpen}
            setValue={setSelectedCondition}
            setItems={setConditions}
            placeholder={''}
            multiple={true}
            style={{
              zIndex: 1,
              borderColor: '#3C6672',
              borderRadius: 15,
            }}
            dropDownContainerStyle={{
              maxHeight: 125,
            }}
          />
        </View>
        <Text
          style={{ ...styles.pickerLabel, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15), elevation: -1 }}
        >
          SELECT YOUR PREFERRED LOCATION
        </Text>
        <View
          style={
            Platform.OS === 'ios'
              ? {
                  marginBottom: 20,
                  width: isLargeScreen ? useResponsiveWidth(35) : useResponsiveWidth(85),
                  zIndex: 1,
                }
              : {
                  marginBottom: 20,
                  width: isLargeScreen ? useResponsiveWidth(35) : useResponsiveWidth(85),
                }
          }
        >
          <DropDownPicker
            open={selectedLocationOpen}
            value={selectedLocation}
            items={locations}
            setOpen={setSelectedLocationOpen}
            setValue={setSelectedLocation}
            setItems={setLocations}
            placeholder={''}
            style={{
              borderColor: '#3C6672',
              borderRadius: 15,
              zIndex: -1,
            }}
            dropDownContainerStyle={{
              maxHeight: 125,
            }}
          />
        </View>
      </View>
      {selectedCondition !== null && selectedLocation !== null ? (
        <View style={{ alignItems: 'center', zIndex: -2, elevation: -2 }}>
          <CustomButton
            isLargeScreen={false}
            styles={styles}
            buttonText={'Next'}
            height={buttonHeight}
            width={buttonWidth}
            onPress={handleNextButtonClicked}
            buttonColor={'#3C6672'}
          />
        </View>
      ) : null}
    </View>
  );
}

function AgreementSection({ styles, isLargeScreen, setState, user }) {
  const dispatch = useDispatch();
  const scrollRef = useRef(null);

  const [scrolledToBottom, setScrolledToBottom] = useState(false);
  const [isChecked, setIsChecked] = useState(false);

  const handleScroll = (event) => {
    const yOffset = event.nativeEvent.contentOffset.y;
    const contentHeight = event.nativeEvent.contentSize.height;
    const scrollViewHeight = event.nativeEvent.layoutMeasurement.height;
    if (yOffset + scrollViewHeight + 25 >= contentHeight) {
      setScrolledToBottom(true);
    }
  };

  const handleSkipToBottom = () => {
    setScrolledToBottom(true);
    scrollRef.current.scrollToEnd({ animated: true });
  };

  const handleChecked = () => {
    setIsChecked(!isChecked);
  };

  const handleNextButtonClicked = () => {
    const eulaStatus = isChecked ? 1 : 0;
    const userID = user.getUserData.userID;
    dispatch(updateEula({ userID, eulaStatus }));
    setState('accountStatus');
  };

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);
  const buttonWidth = isLargeScreen ? useResponsiveWidth(40) : useResponsiveWidth(65);

  const jsonDataEula = require('../data/eula.json');

  return (
    <View>
      <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            LICENSING AGREEMENT
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            To use this software, you must accept the terms of the end user licensing agreement.
          </Text>
        </View>
      </View>
      <View style={{ alignItems: 'center', height: useResponsiveHeight(75), marginTop: 30 }}>
        <Image
          source={primsLogo}
          resizeMode="contain"
          style={{
            width: useResponsiveWidth(55),
            height: useResponsiveHeight(7.5),
            marginBottom: 10,
          }}
        />
        <Text
          style={{
            ...styles.licensingAgreementContentHeader,
            ...textStyles.regular,
            fontSize: useResponsiveFontSize(2.75),
          }}
        >
          User Licensing Agreement
        </Text>
        <View
          style={{
            ...styles.box,
            width: isLargeScreen ? useResponsiveWidth(60) : useResponsiveWidth(90),
            height: useResponsiveHeight(35),
          }}
        >
          <ScrollView
            showsVerticalScrollIndicator={false}
            ref={scrollRef}
            onScroll={handleScroll}
            scrollEventThrottle={15}
            style={styles.scrollContainer}
          >
            <View style={styles.textContent}>
              <Text style={{ fontWeight: 'bold', ...textStyles.bold, marginBottom: 10 }}>User Licensing Agreement</Text>
              {jsonDataEula.map((item) => (
                <Text key={item.id} style={{ marginBottom: 10, ...textStyles.regular }}>
                  {item.Content}
                </Text>
              ))}
            </View>
          </ScrollView>
          {!scrolledToBottom ? (
            <View style={{ paddingRight: 35, paddingLeft: 35 }}>
              <Pressable style={styles.scrollButton} onPress={handleSkipToBottom}>
                <Text style={{ ...styles.scrollButtonText, ...textStyles.regular, fontSize: useResponsiveFontSize(2) }}>
                  Skip to bottom
                </Text>
              </Pressable>
            </View>
          ) : (
            <Pressable onPress={handleChecked}>
              <View style={{ ...styles.checkboxContainer, padding: 20 }}>
                <CheckBox style={{ marginRight: 25 }} value={isChecked} onValueChange={handleChecked} />
                <Text style={{ fontSize: useResponsiveFontSize(2), flexShrink: 1, ...textStyles.regular }}>
                  I accept PRIMS' user licensing agreement
                </Text>
              </View>
            </Pressable>
          )}
        </View>
        {isChecked ? (
          <View style={{ alignItems: 'center' }}>
            <CustomButton
              isLargeScreen={isLargeScreen}
              styles={styles}
              buttonText={'Next'}
              height={buttonHeight}
              width={buttonWidth}
              onPress={handleNextButtonClicked}
              buttonColor={'#3C6672'}
              marginTop={30}
              marginBottom={30}
            />
          </View>
        ) : null}
      </View>
    </View>
  );
}

function AccountTypeSelectionBox({ styles, buttonText, descriptionText, marginTop, marginBottom, onPress, selected }) {
  return (
    <Pressable onPress={onPress}>
      <View
        style={{
          ...styles.container,
          height: useResponsiveWidth(26),
          width: useResponsiveWidth(90),
          marginTop: marginTop,
          marginBottom: marginBottom,
        }}
      >
        <View style={styles.boxContainer}>
          <View
            style={{
              ...styles.smallBox,
              width: useResponsiveWidth(26),
              height: useResponsiveWidth(26),
              backgroundColor: selected ? '#3C6672' : '#D9D9D9',
            }}
          >
            <Text style={{ color: selected ? 'white' : 'black' }}>{buttonText}</Text>
          </View>
          <View style={{ ...styles.textBox, height: useResponsiveWidth(26) }}>
            <Text>{descriptionText}</Text>
          </View>
        </View>
      </View>
    </Pressable>
  );
}
