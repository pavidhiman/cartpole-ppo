// pages/TrialLoginPage.js
import React, { useState } from 'react';
import { Alert, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { useDispatch } from 'react-redux';
import CustomButton from '../components/CustomButton';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { setValidUserMock } from '../slices/ValidUserSlice';

// local list of trial IDs and dummy user objects
const mockUsers = require('../data/mock_users.json');

export default function TrialLoginPage({ navigation }) {
  const dispatch     = useDispatch();
  const isLarge      = useWindowWidth();
  const [trialId, setTrialId] = useState('');

  /** Handle “Continue” */
  const handleLogin = () => {
    if (!trialId.trim()) {
      Alert.alert('Missing Trial ID', 'Please enter a valid ID.');
      return;
    }

    // Find a match in mock_users.json
    const matched = mockUsers.find((u) => u.trialId === trialId.trim());
    if (!matched) {
      Alert.alert('Invalid Trial ID', 'Please try again.');
      return;
    }

    // --- Seed minimal Redux so HomePage doesn’t crash ----------
    const stubUser = {
      userID:    matched.userID  ?? 0,
      patientID: matched.patientID ?? 0,
      firstName: 'Trial',
      lastName:  'Participant',
      userSetupCompleted: 1,
      isEULA: 1,
    };

    // 1) validUser slice
    dispatch(setValidUserMock({ userID: stubUser.userID, patientID: stubUser.patientID }));

    // 2) user slice – fake a fulfilled getUser action
    dispatch({ type: 'user/getUser/fulfilled', payload: stubUser });

    // ------------------------------------------------------------

    // Straight to normal Home
    navigation.reset({ index: 0, routes: [{ name: 'HOME' }] });
  };

  return (
    <View style={{ flex: 1, padding: 24, justifyContent: 'center', backgroundColor: 'white' }}>
      <Text style={{ fontSize: 26, fontWeight: '700', color: '#3C6672', marginBottom: 24 }}>
        Clinical-Trial Login
      </Text>

      <TextInput
        placeholder="Enter Trial ID"
        value={trialId}
        onChangeText={setTrialId}
        autoCapitalize="characters"
        style={{
          borderWidth: 1,
          borderColor: '#3C6672',
          borderRadius: 12,
          padding: 16,
          fontSize: 18,
          marginBottom: 24,
        }}
      />

      <CustomButton
        isLargeScreen={isLarge}
        styles={{}}
        buttonText="Continue"
        height={60}
        width="100%"
        buttonColor="#3C6672"
        onPress={handleLogin}
      />

      <TouchableOpacity onPress={() => navigation.goBack()} style={{ marginTop: 20 }}>
        <Text style={{ color: '#3C6672', textAlign: 'center' }}>
          Back to regular login
        </Text>
      </TouchableOpacity>
    </View>
  );
}
