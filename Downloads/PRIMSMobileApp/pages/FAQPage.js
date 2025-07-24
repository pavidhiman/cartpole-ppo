// Components
import { View, StyleSheet, Text, ScrollView, TouchableOpacity } from 'react-native';
import DownArrow from '../components/svg/DownArrow';
import UpArrow from '../components/svg/UpArrow';
import HorizontalRule from '../components/HorizontalRule';

// Hooks
import { useResponsiveHeight, useResponsiveWidth, useResponsiveFontSize } from 'react-native-responsive-dimensions';
import { useState } from 'react';

// Styles
import { textStyles } from '../styles/textStyles';
import { settingsPageStyles } from '../styles/SettingsPageStyles';

// Data
import * as data from '../data/faq.json';

export default function FAQPage() {
  const questions = data.default;
  const questionTextSize = useResponsiveFontSize(2);
  const answerTextSize = useResponsiveFontSize(1.8);

  const [questionStatus, setQuestionStatus] = useState(Array(questions.length).fill(false));

  const handlePress = (index) => {
    setQuestionStatus((prev) => {
      const newState = [...prev];
      newState[index] = !newState[index];
      return newState;
    });
  };

  return (
    <View style={styles.wrapper}>
      <View style={{ ...settingsPageStyles.topContainer, height: useResponsiveHeight(25) }}>
        <View style={{ width: useResponsiveWidth(75) }}>
          <Text style={{ ...settingsPageStyles.mainHeader, ...textStyles.regular, fontSize: useResponsiveFontSize(3) }}>
            FREQUENTLY ASKED QUESTIONS
          </Text>
          <Text
            style={{ ...settingsPageStyles.mainText, ...textStyles.regular, fontSize: useResponsiveFontSize(2.15) }}
          >
            View the answers to some frequently asked questions
          </Text>
        </View>
      </View>
      <View style={styles.scrollWrapper}>
        <ScrollView style={styles.scroll}>
          {questions.map((question, index) => {
            return (
              <View key={index} style={{ marginBottom: 35 }}>
                <TouchableOpacity
                  style={styles.question}
                  onPress={() => {
                    handlePress(index);
                  }}
                >
                  <Text style={{ ...textStyles.regular, fontSize: questionTextSize }}>{question.question}</Text>
                  {questionStatus[index] ? <UpArrow /> : <DownArrow />}
                </TouchableOpacity>
                {questionStatus[index] && (
                  <Text style={{ ...textStyles.regular, fontSize: answerTextSize, marginTop: 35 }}>
                    {question.answer}
                  </Text>
                )}
                {index !== questionStatus.length - 1 && (
                  <View>
                    <View style={{ height: 35 }} />
                    <HorizontalRule color={'#E8E8E8'} />
                  </View>
                )}
              </View>
            );
          })}
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { flex: 1 },
  scrollWrapper: { flex: 1, paddingTop: 45, paddingBottom: 125, paddingLeft: 45, paddingRight: 45 },
  scroll: { flex: 1 },
  question: { flexDirection: 'row', justifyContent: 'space-between' },
});
