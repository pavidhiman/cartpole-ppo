// Components
import { Image, ScrollView, Text, View } from 'react-native';
import HorizontalRule from '../components/HorizontalRule';

// Hooks
import { useResponsiveFontSize, useResponsiveHeight, useResponsiveWidth } from 'react-native-responsive-dimensions';

// Assets
import clinicianInfo from '../assets/images/clinician-info.png';
import dataCollection from '../assets/images/data-collection.png';
import patientRatings from '../assets/images/patient-ratings.png';
import primsSystems from '../assets/images/prims-system.png';

// Styles
import { useEffect } from 'react';
import { settingsPageStyles } from '../styles/SettingsPageStyles';
import { textStyles } from '../styles/textStyles';

import * as Amplitude from '@amplitude/analytics-react-native';

export default function AboutPage() {
  const largeText = useResponsiveFontSize(2);
  const smallText = useResponsiveFontSize(1.8);

  useEffect(() => {
    Amplitude.logEvent('OPEN_ABOUT_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_ABOUT_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  return (
    <View style={{ flex: 1 }}>
      <View style={{ ...settingsPageStyles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...settingsPageStyles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            ABOUT
          </Text>
          <Text
            style={{ ...settingsPageStyles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}
          >
            Learn more about the PRIMS software
          </Text>
        </View>
      </View>
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={{ padding: 25, paddingBottom: 105, flex: 1 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Image source={primsSystems} style={{ width: 58, height: 50 }} />
            <View style={{ flex: 1, paddingLeft: 12 }}>
              <Text style={{ ...textStyles.regular, fontSize: largeText }}>What is our PRIMS system?</Text>
              <View style={{ height: 7 }}></View>
              <HorizontalRule color={'#9FCEAE'} />
              <View style={{ height: 7 }}></View>
              <Text style={{ ...textStyles.regular, fontSize: smallText }}>
                Our first patient assessment tool called PRIMS, is a proprietary out-of-the-box integrated system that
                will monitor, analyze and rate the severity of a full range of symptoms. Currently, the system analyzes
                Parkinson's symptoms.
              </Text>
            </View>
          </View>
          <View style={{ height: 35 }} />
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Image source={dataCollection} style={{ width: 58, height: 50 }} />
            <View style={{ flex: 1, paddingLeft: 12 }}>
              <Text style={{ ...textStyles.regular, fontSize: largeText }}>Data Collection</Text>
              <View style={{ height: 7 }}></View>
              <HorizontalRule color={'#3C6672'} />
              <View style={{ height: 7 }}></View>
              <Text style={{ ...textStyles.regular, fontSize: smallText }}>
                The PRIMS system collects real-time data through the system without the need for sensors on their body.
                The data collected will then be available to the patient, neurologist, and physiotherapists whenever
                they want to access this information.
              </Text>
            </View>
          </View>
          <View style={{ height: 35 }} />
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Image source={patientRatings} style={{ width: 58, height: 50 }} />
            <View style={{ flex: 1, paddingLeft: 12 }}>
              <Text style={{ ...textStyles.regular, fontSize: largeText }}>Accurate Patient Ratings</Text>
              <View style={{ height: 7 }}></View>
              <HorizontalRule color={'#84A258'} />
              <View style={{ height: 7 }}></View>
              <Text style={{ ...textStyles.regular, fontSize: smallText }}>
                Once the assessments have been completed, the software will give an accurate rating based on real-time
                data and put the patient on a Parkinson's rating scale from 0 - 4. These results will also be assessed
                by a Clinician.
              </Text>
            </View>
          </View>
          <View style={{ height: 35 }} />
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Image source={clinicianInfo} style={{ width: 58, height: 50 }} />
            <View style={{ flex: 1, paddingLeft: 12 }}>
              <Text style={{ ...textStyles.regular, fontSize: largeText }}>Information for Clinicians</Text>
              <View style={{ height: 7 }}></View>
              <HorizontalRule color={'#DEBE5B'} />
              <View style={{ height: 7 }}></View>
              <Text style={{ ...textStyles.regular, fontSize: smallText }}>
                The data collected from the PRIMS software will help Clinicians assess and monitor a patient's condition
                which will allow them to adjust their medication if needed. The system will assist Clinicians in better
                treatment of their Parkinson's Disease patients.
              </Text>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}
