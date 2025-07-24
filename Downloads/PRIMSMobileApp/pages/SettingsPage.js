import * as Amplitude from '@amplitude/analytics-react-native';
import CheckBox from 'expo-checkbox';
import { useEffect, useState } from 'react';
import { Image, Pressable, Switch, Text, TextInput, View } from 'react-native';
import DropDownPicker from 'react-native-dropdown-picker';
import { useResponsiveFontSize, useResponsiveHeight, useResponsiveWidth } from 'react-native-responsive-dimensions';
import { useDispatch, useSelector } from 'react-redux';
import gif from '../assets/images/logo-loader-whitebg.gif';
import minusImg from '../assets/images/minus.png';
import plusImg from '../assets/images/plus.png';
import CustomButton from '../components/CustomButton';
import Dropdown from '../components/Dropdown';
import Popup from '../components/Popup/Popup';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { getConditions } from '../slices/ConditionsSlice';
import { getLocations } from '../slices/LocationsSlice';
import { getPatientSettings, savePatientSettings } from '../slices/PatientSettingsSlice';
import { settingsPageStyles } from '../styles/SettingsPageStyles';
import { textStyles } from '../styles/textStyles';
import patientSettingsData from '../data/patientSettings.json';
import { Feather } from '@expo/vector-icons'; // Add this import for pen/check icons
// Toggle to use JSON for settings (for testing)
const USE_JSON_FOR_SETTINGS = true;
let jsonSettings = null;
if (USE_JSON_FOR_SETTINGS) {
  jsonSettings = require('../data/patientSettings.json');
  if (!jsonSettings.notificationSettings) {
    jsonSettings.notificationSettings = {
      userNotificationSettingsID: 1,
      userID: 1,
      receiveNotifications: 1,
      emailNotices: 1,
      textNotices: 1,
      appNotices: 1,
    };
  }
  if (!jsonSettings.patientConditions) {
    jsonSettings.patientConditions = [
      { patientID: jsonSettings.patient.patientID, conditionID: 1 }
    ];
  }
}

export default function SettingsPage({ navigation, setHasUnsavedChanges }) {
  const styles = settingsPageStyles;

  const isLargeScreen = useWindowWidth();

  // Use JSON data directly for initial state
  const [firstName, setFirstName] = useState(patientSettingsData.patient.firstName || '');
  const [lastName, setLastName] = useState(patientSettingsData.patient.lastName || '');
  const [dob, setDob] = useState(patientSettingsData.patient.dob || '');
  const [gender, setGender] = useState(patientSettingsData.patient.gender || '');
  const patientConditions = patientSettingsData.patientConditions || [];
  const notificationSettings = patientSettingsData.notificationSettings || {};

  // Edit state for each field
  const [editField, setEditField] = useState(null); // 'firstName', 'lastName', 'dob', 'gender', or null

  // Helper to render a demographic row
  const renderDemographicRow = (label, value, field, setter) => (
    <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 18, backgroundColor: '#fff', borderRadius: 12, padding: 16, shadowColor: '#000', shadowOpacity: 0.06, shadowRadius: 4, elevation: 2 }}>
      <Text style={{ flex: 1, fontSize: 17, fontWeight: '500', color: '#2d3a4a' }}>{label}</Text>
      {editField === field ? (
        <>
          <TextInput
            style={{ flex: 2, fontSize: 17, borderBottomWidth: 1, borderColor: '#3C6672', marginRight: 10, paddingVertical: 2 }}
            value={value}
            onChangeText={setter}
            autoFocus
          />
          <Pressable onPress={() => setEditField(null)}>
            <Feather name="check" size={22} color="#3C6672" />
          </Pressable>
        </>
      ) : (
        <>
          <Text style={{ flex: 2, fontSize: 17, color: '#3C6672', marginRight: 10 }}>{value}</Text>
          <Pressable onPress={() => setEditField(field)}>
            <Feather name="edit-2" size={20} color="#3C6672" />
          </Pressable>
        </>
      )}
    </View>
  );

  const [patientCurrentConditions, setPatientCurrentConditions] = useState(patientConditions);
  const [conditionToRemove, setConditionToRemove] = useState(null);

  const [notificationsToggled, setNotificationsToggled] = useState(
    notificationSettings.emailNotices === 1 && notificationSettings.textNotices === 1
  );
  const [textNotificationsToggled, setTextNotificationsToggled] = useState(
    notificationSettings.textNotices === 1
  );
  const [emailNotificationsToggled, setEmailNotificationsToggled] = useState(
    notificationSettings.emailNotices === 1
  );
  const [pushNotificationsToggled, setPushNotificationsToggled] = useState(
    notificationSettings.appNotices === 1
  );

  const [showUpdatedPopup, setShowUpdatedPopup] = useState(false);
  const [showRemovePopup, setShowRemovePopup] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(false);

  const conditionsArray = patientSettingsData.conditions || [];

  const locationToID = patientSettingsData.locations || {};

  const patientConditionsIDArray = patientCurrentConditions.map((patient) => {
    const matchingCondition = conditionsArray.find((condition) => condition.condition === patient.condition);
    if (matchingCondition) {
      return {
        patientID: patient.patientID,
        conditionID: matchingCondition.conditionID,
      };
    }
  });

  useEffect(() => {
    if (settingsSaved) {
      navigation.navigate('HOME');
    }
  }, [settingsSaved]);

  useEffect(() => {
    const data = {
      notificationSettings: {
        userNotificationSettingsID: notificationSettings.userNotificationSettingsID,
        userID: patientSettingsData.user?.userID || 1,
        receiveNotifications: notificationsToggled ? 1 : 0,
        emailNotices: emailNotificationsToggled ? 1 : 0,
        textNotices: textNotificationsToggled ? 1 : 0,
        appNotices: pushNotificationsToggled ? 1 : 0,
      },
    };
    if (settingsSaved) {
      return;
    }
    const hasUnsavedChanges =
      JSON.stringify(patientConditionsIDArray) !== JSON.stringify(patientConditionsIDArray) ||
      JSON.stringify(notificationSettings) !== JSON.stringify(data.notificationSettings);
    setHasUnsavedChanges(hasUnsavedChanges);
  }, [
    notificationsToggled,
    emailNotificationsToggled,
    textNotificationsToggled,
    pushNotificationsToggled,
    patientCurrentConditions,
    settingsSaved,
  ]);

  const rowWidth = useResponsiveWidth(80);
  const conditionTextSize = useResponsiveFontSize(1.85);
  const removeBtnHeight = useResponsiveHeight(0.75);
  const removeBtnWidth = useResponsiveWidth(6.5);
  const buttonFontSize = useResponsiveFontSize(1.85);

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

  const handleConfirmBtnPressed = () => {
    setShowUpdatedPopup(false);
    setSettingsSaved(true);
  };

  const removeConditionSelected = (condition) => {
    if (patientCurrentConditions.length === 1 && patientCurrentConditions[0].condition === 'Unknown') {
      return;
    }
    setConditionToRemove(condition);
    setShowRemovePopup(true);
  };

  const completeRemoveCondition = () => {
    let newCurrentConditions = patientCurrentConditions.filter((item) => item.condition !== conditionToRemove);
    if (newCurrentConditions.length === 0) {
      newCurrentConditions = [{ 'condition': 'Unknown', 'patientID': patientSettingsData.patient.patientID }];
    }
    setPatientCurrentConditions(newCurrentConditions);
    setShowRemovePopup(false);
  };

  const handleAddAnotherClicked = () => {
    setShowDropdown(!showDropdown);
  };

  const handleNoBtnPressed = () => {
    setShowRemovePopup(false);
  };

  const addNewCondition = (selection) => {
    if (selection !== 'Unknown') {
      const newCondition = { condition: selection, patientID: patientSettingsData.patient.patientID };
      if (
        !patientCurrentConditions.some(
          (condition) =>
            condition.condition === newCondition.condition && condition.patientID === newCondition.patientID,
        )
      ) {
        const updatedConditions = patientCurrentConditions.filter((item) => item.condition !== 'Unknown');

        updatedConditions.push(newCondition);

        setPatientCurrentConditions(updatedConditions);
        setShowDropdown(!showDropdown);
      }
    }
  };

  const handleSaveBtnClicked = () => {
    const data = {
      patientId: patientSettingsData.patient.patientID,
      patientConditions: patientConditionsIDArray,
      conditions: conditionsArray,
      patient: {
        patientID: patientSettingsData.patient.patientID,
        userID: patientSettingsData.user?.userID || 1,
        firstName,
        lastName,
        dob,
        gender,
        conditions: '',
        lastAssessmentDate: patientSettingsData.user?.dob,
      },
      notificationSettings: {
        userNotificationSettingsID: notificationSettings.userNotificationSettingsID,
        userID: patientSettingsData.user?.userID || 1,
        receiveNotifications: notificationsToggled ? 1 : 0,
        emailNotices: emailNotificationsToggled ? 1 : 0,
        textNotices: textNotificationsToggled ? 1 : 0,
        appNotices: pushNotificationsToggled ? 1 : 0,
      },
    };
    // dispatch(savePatientSettings({ data: data })); // Removed Redux dispatch
    setShowUpdatedPopup(true);
  };

  const btnHeightLarge = useResponsiveHeight(12);
  const btnHeightSmall = useResponsiveHeight(7);

  const popupMarginBottom = useResponsiveHeight(25);

  const patientConditionsArray = [];
  for (let condition of patientCurrentConditions) {
    patientConditionsArray.push({ patientID: patientSettingsData.patient.patientID, conditionID: condition.conditionID });
  }

  const dropdownOptions = [];
  for (let condition of conditionsArray) {
    dropdownOptions.push(condition.condition);
  }

  return (
    <View style={[styles.container, { backgroundColor: '#f6f8fa', flex: 1, padding: 0 }]}>
      <TopHeader styles={styles} />
      {showUpdatedPopup ? (
        <Popup
          header={'Updated Settings'}
          text={`Your settings have been updated.`}
          buttonAmount={1}
          firstButtonText={'Confirm'}
          leftFunction={handleConfirmBtnPressed}
          coloredButton={'left'}
          theme={'general'}
          isLargeScreen={isLargeScreen}
          singleButtonWidth={60}
          marginBottom={popupMarginBottom}
        />
      ) : null}
      {showRemovePopup ? (
        <Popup
          header={'Remove Condition'}
          text={`Are you sure you want to remove the selected condition?`}
          buttonAmount={2}
          firstButtonText={'No'}
          secondButtonText={'Yes'}
          leftFunction={handleNoBtnPressed}
          rightFunction={completeRemoveCondition}
          coloredButton={'right'}
          theme={'general'}
          isLargeScreen={isLargeScreen}
          marginBottom={popupMarginBottom}
        />
      ) : null}
      <View style={{ margin: 24, backgroundColor: '#fff', borderRadius: 16, padding: 20, shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 }}>
        <Text style={{ fontSize: 22, fontWeight: '700', color: '#3C6672', marginBottom: 18, letterSpacing: 0.5 }}>
          Demographic Information
        </Text>
        {renderDemographicRow('First Name', firstName, 'firstName', setFirstName)}
        {renderDemographicRow('Last Name', lastName, 'lastName', setLastName)}
        {renderDemographicRow('Date of Birth', dob, 'dob', setDob)}
        {renderDemographicRow('Gender', gender, 'gender', setGender)}
      </View>
      <View style={{ marginHorizontal: 24, marginTop: 10, backgroundColor: '#fff', borderRadius: 16, padding: 20, shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 8, elevation: 3 }}>
        <Text style={{ fontSize: 22, fontWeight: '700', color: '#3C6672', marginBottom: 18, letterSpacing: 0.5 }}>
          Conditions
        </Text>
        <View style={{ marginTop: 10 }}>
          {patientConditions.length > 0 ? (
            patientConditions.map((cond, idx) => (
              <Text key={idx} style={{ fontSize: 16, color: '#2d3a4a', marginBottom: 8 }}>
                • Condition ID: {cond.conditionID}
              </Text>
            ))
          ) : (
            <Text style={{ fontSize: 16, color: '#2d3a4a' }}>No conditions listed.</Text>
          )}
        </View>
      </View>
      {/* Add more sections/cards as needed */}
    </View>
  );
}

function TopHeader({ styles }) {
  return (
    <View style={{ ...styles.topContainer, height: useResponsiveHeight(25) }}>
      <View style={{ width: useResponsiveWidth(75) }}>
        <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
          SETTINGS
        </Text>
        <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
          View and modify account settings and profile preferences.
        </Text>
      </View>
    </View>
  );
}
