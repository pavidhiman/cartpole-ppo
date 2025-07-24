// components
import { Dimensions, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import HorizontalRule from '../components/HorizontalRule';
import SlidingView from '../components/SlidingView';

// PRIMS API
import { getOneYrSurveys, getPatientSurveyAnswerList } from '../slices/SurveySlice';

// hooks
import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

// styles
import { textStyles } from '../styles/textStyles';

import * as Amplitude from '@amplitude/analytics-react-native';

export default function SurveyHistoryPage() {
  // redux
  const surveys = useSelector((state) => state.surveys);
  const validUser = useSelector((state) => state.validUser);

  const dispatch = useDispatch();

  const [openSlider, setOpenSlider] = useState(true);
  const [surveySelected, setSurveySelected] = useState(false);
  const [currentSurvey, setCurrentSurvey] = useState({
    name: null,
    date: null,
    answers: [],
  });
  const [selectedSurveyIndex, setSelectedSurveyIndex] = useState(null);

  const getSurveyHistory = () => {
    dispatch(getOneYrSurveys({ patientID: validUser.data.patientID }));
  };

  const handleSurveySelect = (survey) => {
    setSurveySelected(true);
    setCurrentSurvey({
      name: survey.surveyDesc,
      date: new Date(survey.dateTimeCompleted).toLocaleString('default', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      }),
      answers: survey.isClinicalTrial ? [] : [],
      isClinicalTrial: survey.isClinicalTrial,
      patientSurveyID: survey.patientSurveyID,
    });
    if (!survey.isClinicalTrial) {
      dispatch(getPatientSurveyAnswerList({ patientSurveyID: survey.patientSurveyID }));
    }
  };

  const handleSurveyConfirm = () => {
    setOpenSlider(false);
    setSurveySelected(false);
    setSelectedSurveyIndex(null);
  };

  useEffect(() => {
    getSurveyHistory();
  }, []);

  useEffect(() => {
    if (surveys.getPatientSurveyAnswerListData && surveys.getPatientSurveyAnswerListData.length > 0) {
      // Find the survey data that matches the selected survey
      const surveyData = surveys.getPatientSurveyAnswerListData.find(
        survey => survey.patientSurveyID === currentSurvey.patientSurveyID
      );
      
      if (surveyData && surveyData.answers) {
        setCurrentSurvey(prev => ({
          ...prev,
          answers: surveyData.answers
        }));
      }
    }
  }, [surveys.getPatientSurveyAnswerListData, currentSurvey.patientSurveyID]);

  useEffect(() => {
    Amplitude.logEvent('OPEN_SURVEY_HISTORY_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_SURVEY_HISTORY_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  // Filter out clinical trial surveys
  const availableSurveys = Array.isArray(surveys.getOneYrSurveysData) 
    ? surveys.getOneYrSurveysData
    : [];

  return (
    <View style={surveyHistoryStyles.wrapper}>
      <SlidingView
        heading="SURVEY HISTORY"
        color="#3C6672"
        isOpen={openSlider}
        onOpen={() => {
          setOpenSlider(true);
        }}
        style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}
      >
        {
          <View //THIS IS THE HISTORY WRAPPER
            style={{
              ...sliderStyles.surveysWrapper,
              flex: 8,
            }}
          >
            <ScrollView showsVerticalScrollIndicator={false}>
            {/* show a single line when nothing came back */}
            {availableSurveys.length === 0 && (
              <Text style={{ ...textStyles.regular, textAlign: 'center', marginTop: 20 }}>
                No surveys completed in the last 12 months.
              </Text>
            )}

            {availableSurveys.map((survey, i) => {
              return (
                <Pressable
                  key={i}
                  style={() => {
                    return i === selectedSurveyIndex
                      ? { ...sliderStyles.survey, backgroundColor: '#2B4851' }
                      : sliderStyles.survey;
                  }}
                  onPress={() => {
                    handleSurveySelect(survey);
                    setSelectedSurveyIndex(i);
                  }}
                >
                  <View>
                    <Text style={{ ...textStyles.bold, fontSize: 16, color: 'white' }}>{survey.surveyDesc}</Text>
                    <Text style={{ ...textStyles.regular, fontSize: 16, color: 'white' }}>
                      {new Date(survey.dateTimeCompleted).toLocaleString('default', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </Text>
                    {survey.isClinicalTrial && (
                      <Text style={{ ...textStyles.regular, fontSize: 12, color: '#FFD700', marginTop: 5 }}>
                        Clinical Trial
                      </Text>
                    )}
                  </View>
                </Pressable>
              );
            })}
            </ScrollView>
          </View>
        }
        <View style={{ ...sliderStyles.confirmWrapper, flex: 3, paddingBottom: 25 }}>
          {surveySelected && (
            <Pressable onPress={handleSurveyConfirm}>
              {console.log('button active')}
              <View style={sliderStyles.confirm}>
                <Text style={{ ...textStyles.regular, fontSize: 14 }}>Confirm</Text>
              </View>
            </Pressable>
          )}
        </View>
      </SlidingView>

      <View style={surveyHistoryStyles.summary}>
        <View>
          <Text style={textStyles.bold}>{currentSurvey.name}</Text>
          <Text style={textStyles.regular}>{currentSurvey.date}</Text>
        </View>
      </View>

      <View style={surveyHistoryStyles.legend}>
        <Text style={{ ...textStyles.regular, fontSize: 10, color: '#5E5E5E' }}>Question</Text>
        <Text style={{ ...textStyles.regular, fontSize: 10, color: '#5E5E5E' }}>Answer</Text>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} style={surveyHistoryStyles.results}>
        {currentSurvey.isClinicalTrial ? (
          <View style={{ padding: 20, alignItems: 'center' }}>
            <Text style={{ ...textStyles.bold, fontSize: 18, marginBottom: 10, color: '#3C6672' }}>
              Clinical Trial Survey
            </Text>
            <Text style={{ ...textStyles.regular, fontSize: 14, textAlign: 'center', color: '#666' }}>
              This survey is part of a clinical trial. The detailed results are not available to patients for privacy and research purposes.
            </Text>
          </View>
        ) : (
          currentSurvey.answers && currentSurvey.answers.length > 0 ? (
            currentSurvey.answers.map((answer, i) => {
              return (
                <View key={i}>
                  <View style={surveyHistoryStyles.question}>
                    <Text style={{ ...textStyles.regular, width: 200 }}>{`${i + 1}. ${answer.componentTitle}`}</Text>
                    <Text style={textStyles.regular}>{answer.answerText}</Text>
                  </View>
                  {i !== currentSurvey.answers.length - 1 && <HorizontalRule color="#E8E8E8" />}
                </View>
              );
            })
          ) : (
            <View style={{ padding: 20, alignItems: 'center' }}>
              <Text style={{ ...textStyles.regular, fontSize: 14, textAlign: 'center', color: '#666' }}>
                Loading survey details...
              </Text>
            </View>
          )
        )}
      </ScrollView>
    </View>
  );
}

const sliderStyles = StyleSheet.create({
  surveysWrapper: {
    height: Dimensions.get('window').height * (2 / 3),
    paddingTop: 57,
    paddingRight: 13,
    paddingLeft: 13,
  },

  survey: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 18,
    borderWidth: 1,
    borderRadius: 12,
    borderColor: '#2B4851',
    marginBottom: 9,
  },

  confirmWrapper: { alignItems: 'center', marginTop: 18 },

  confirm: {
    height: 40,
    width: 185,
    borderRadius: 12,
    backgroundColor: '#D9D9D9',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

const surveyHistoryStyles = StyleSheet.create({
  wrapper: {
    flex: 1,
    paddingTop: 90,
    paddingRight: 9,
    paddingLeft: 9,
    paddingBottom: 80,
    gap: 11,
    zIndex: 0,
  },

  summary: {
    height: 46,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingLeft: 27,
    paddingRight: 27,
  },

  legend: {
    backgroundColor: '#F3F3F3',
    height: 46,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E8E8E8',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 33,
    paddingLeft: 33,
  },

  results: { flex: 1, borderRadius: 12, borderWidth: 1, borderColor: '#E8E8E8', paddingLeft: 33, paddingRight: 33 },

  question: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', height: 90 },
});
