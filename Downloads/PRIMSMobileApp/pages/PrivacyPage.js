import { Text, View, ScrollView, Image } from 'react-native';
import React, { useState, useEffect } from 'react';
import { privacyPageStyles } from '../styles/PrivacyPageStyles';
import { textStyles } from '../styles/textStyles';
import primsLogo from '../assets/images/PRIMS-Logo.png';
const jsonDataPrivacy = require('../data/privacy.json');

import { useResponsiveHeight, useResponsiveWidth, useResponsiveFontSize } from 'react-native-responsive-dimensions';

import { useWindowWidth } from '../hooks/useWindowWidth';

import * as Amplitude from '@amplitude/analytics-react-native';

export default function PrivacyPage() {
  const isLargeScreen = useWindowWidth();

  const styles = privacyPageStyles;

  let topContainerHeight = useResponsiveHeight(25);

  let mediumHeader = useResponsiveFontSize(2.15);

  useEffect(() => {
    Amplitude.logEvent('OPEN_PRIVACY_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_PRIVACY_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  return (
    <View style={styles.pageContainer}>
      <View style={{ ...styles.topContainer, height: topContainerHeight }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            PRIVACY
          </Text>
          <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}>
            Learn more about how we collect and use your data.
          </Text>
        </View>
      </View>
      <View style={{ ...styles.privacyHeader, height: useResponsiveHeight(15) }}>
        <Image
          source={primsLogo}
          style={{ width: useResponsiveWidth(40), height: useResponsiveWidth(12), marginBottom: 10 }}
        />
        <Text style={{ ...textStyles.regular, fontSize: useResponsiveFontSize(2.5) }}>Privacy Policy</Text>
      </View>
      <View style={{ alignItems: 'center' }}>
        <View
          style={{
            ...styles.box,
            width: isLargeScreen ? useResponsiveWidth(60) : useResponsiveWidth(90),
            height: useResponsiveHeight(40),
          }}
        >
          <ScrollView
            showsVerticalScrollIndicator={false}
            style={{
              width: useResponsiveWidth(90),
            }}
          >
            {jsonDataPrivacy.map((item) => (
              <View key={item.id} style={{ padding: 10 }}>
                <Text style={{ fontSize: mediumHeader, marginBottom: 5 }}>{item.Heading}</Text>
                <Text style={{ marginBottom: 10 }}>{item.Content}</Text>
              </View>
            ))}
          </ScrollView>
        </View>
      </View>
    </View>
  );
}
