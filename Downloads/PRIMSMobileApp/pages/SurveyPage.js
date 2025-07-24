import { Text, View, Pressable, Image } from 'react-native';
import React, { useState, useEffect } from 'react';
import { useResponsiveHeight, useResponsiveWidth, useResponsiveFontSize } from 'react-native-responsive-dimensions';
import { surveyPageStyles } from '../styles/SurveyPageStyles';
import PRIMSLogo from '../assets/images/PRIMS-Logo.png';
import { useWindowWidth } from '../hooks/useWindowWidth';
import { useSelector, useDispatch } from 'react-redux';
import { textStyles } from '../styles/textStyles';
import BottomNavbar from '../components/BottomNavbar';
import Popup from '../components/Popup/Popup';
import gif from '../assets/images/logo-loader-whitebg.gif';
import * as Amplitude from '@amplitude/analytics-react-native';

import { saveSurveyDetails, saveSurveyAnswers, getLatestSurvey } from '../slices/SurveySlice';

const surveyStart = new Date().toISOString().slice(0, 19);

export default function SurveyPage({ navigation }) {
  const isLargeScreen = useWindowWidth();

  const styles = surveyPageStyles;

  const surveys = useSelector((state) => state.surveys);
  const validUser = useSelector((state) => state.validUser);

  const dispatch = useDispatch();

  // Defensive: Only use surveys.getSurveyQuestionsData if it's a non-empty array
  const questionsData = Array.isArray(surveys.getSurveyQuestionsData) ? surveys.getSurveyQuestionsData : [];
  const answersData = Array.isArray(surveys.getSurveyAnswersData) ? surveys.getSurveyAnswersData : [];

  const [questionIndex, setQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState(() => {
    let answersTemplate = [];
    for (let i = 0; i < questionsData.length; ++i) {
      answersTemplate.push({
        patientSurveyID: '',
        componentID: '',
        answerID: '',
      });
    }
    return answersTemplate;
  });

  const [toggleNextButton, setToggleNextButton] = useState(false);

  const [activeAnswer, setActiveAnswer] = useState('');
  const [nextButtonText, setNextButtonText] = useState('Next');
  const [donePopup, setDonePopup] = useState(false);
  const [surveyDone, setSurveyDone] = useState(false);

  // console.log('USERANSWERS ***', userAnswers);

  // Defensive: Only reduce if answersData is an array
  const groupedAnswers = answersData.reduce((result, currentItem) => {
    const key = currentItem.componentID;

    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(currentItem);

    return result;
  }, {});

  const handleSelect = (selection, componentID) => {
    setUserAnswers((prev) => {
      let temp = [...prev];
      temp[questionIndex]['answerID'] = selection;
      temp[questionIndex]['componentID'] = componentID;
      setActiveAnswer(selection);

      return temp;
    });
    setToggleNextButton(true);
  };

  const handleNext = () => {
    if (nextButtonText === 'Next') {
      setToggleNextButton(false);
    }

    if (questionIndex + 1 < questionsData.length) {
      setQuestionIndex((prev) => {
        let temp = prev + 1;
        return temp;
      });
    }

    if (questionIndex + 2 === questionsData.length) {
      setNextButtonText('Complete');
    }

    if (nextButtonText === 'Complete') {
      toggleDonePopup();
    }

    if (nextButtonText === 'Done') {
      navigation.navigate('HOME');
    }
  };

  const toggleDonePopup = () => {
    setDonePopup(!donePopup);
  };

  const handleSaveSurveyDetails = async () => {
    const data = {
      patientID: validUser.data.patientID,

      completedByUserID: validUser.data.userID,

      surveyID: answersData[0].surveyID,

      dateTimeStarted: surveyStart, //Wrong time, need to figure out later

      dateTimeCompleted: new Date().toISOString().slice(0, 19),

      locationID: 100,

      surveyScore: null,

      primaryInformationSource: 'Self',

      discussion: null,

      notes: null,

      followup: null,

      uUID: null,
    };
    dispatch(saveSurveyDetails({ data: data }));

    setSurveyDone(!surveyDone);
    setDonePopup(!donePopup);
    setNextButtonText('Done');
    setToggleNextButton(false);
  };

  useEffect(() => {
    if (surveys.saveSurveyDetailsStatus === 'succeeded') {
      const answersData = userAnswers.map((answer) => ({
        'answerID': answer.answerID,
        'componentID': answer.componentID,
        'patientSurveyID': surveys.saveSurveyDetailsData['patientSurveyID'],
      }));

      dispatch(saveSurveyAnswers({ data: answersData }));
    }
  }, [surveys.saveSurveyDetailsStatus]);

  useEffect(() => {
    if (surveys.saveSurveyAnswersStatus === 'succeeded') {
      dispatch(getLatestSurvey({ patientID: validUser.data.patientID }));
    }
  }, [surveys.saveSurveyAnswersStatus]);

  useEffect(() => {
    if (surveys.getLatestSurveyStatus === 'succeeded') {
      setToggleNextButton(true);
    }
  }, [surveys.getLatestSurveyStatus]);

  useEffect(() => {
    Amplitude.logEvent('OPEN_SURVEY_PAGE');
    const startTime = new Date();

    return () => {
      const duration = (new Date() - startTime) / 1000;
      Amplitude.logEvent('CLOSE_SURVEY_PAGE', {
        'duration': duration,
      });
    };
  }, []);

  const buttonHeight = isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(7);
  const buttonWidth = useResponsiveWidth(65);
  const fontSize = isLargeScreen ? useResponsiveFontSize(1.85) : useResponsiveFontSize(2.5);
  const topContainerHeight = useResponsiveHeight(25);
  const topContainerWidthQuestions = useResponsiveWidth(90);
  const topContainerWidthResults = useResponsiveWidth(75);
  const mainHeaderFontSizeQuestions = useResponsiveFontSize(2.5);
  const mainHeaderFontSizeResults = useResponsiveFontSize(3);
  const bodyFontSize = useResponsiveFontSize(2.15);
  const loaderWidth = useResponsiveWidth(25);
  const loaderHeight = useResponsiveWidth(25);

  return (
    <View style={styles.pageContainer}>
      {!surveyDone && questionsData.length > 0 && (
        <View style={{ ...styles.topContainer, height: topContainerHeight }}>
          <View style={{ width: topContainerWidthQuestions }}>
            {questionsData[questionIndex] ? (
              <>
                <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: mainHeaderFontSizeQuestions }}>
                  {questionsData[questionIndex].componentNumber}
                  {'. '}
                  {questionsData[questionIndex].componentTitle}
                </Text>
                <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: bodyFontSize }}>
                  {questionsData[questionIndex].componentDesc}
                </Text>
              </>
            ) : (
              <Text style={{ ...textStyles.regular, fontSize: bodyFontSize }}>
                No question available.
              </Text>
            )}
          </View>
        </View>
      )}
      {/* Optionally, add a fallback UI if questionsData is empty */}
      {!surveyDone && questionsData.length === 0 && (
        <View style={{ alignItems: 'center', marginTop: topContainerHeight, flex: 1 }}>
          <Text style={{ ...textStyles.regular, fontSize: bodyFontSize }}>
            No survey questions available.
          </Text>
        </View>
      )}
      {surveyDone && (
        <View style={{ ...styles.topContainer, height: topContainerHeight }}>
          <View style={{ width: topContainerWidthResults }}>
            <Text style={{ ...styles.mainHeader, ...textStyles.regular, fontSize: mainHeaderFontSizeResults }}>
              SURVEY COMPLETE
            </Text>
            <Text style={{ ...styles.mainText, ...textStyles.regular, fontSize: bodyFontSize }}>
              Your survey has been scored and submitted.
            </Text>
          </View>
        </View>
      )}
      {donePopup && (
        <Popup
          text={'Are you sure you want to submit your survey?'}
          header={'Submit'}
          buttonAmount={2}
          firstButtonText={'No'}
          secondButtonText={'Yes'}
          coloredButton={'right'}
          theme={'general'}
          rightFunction={handleSaveSurveyDetails}
          leftFunction={toggleDonePopup}
          isLargeScreen={isLargeScreen}
        ></Popup>
      )}

      <View style={styles.mainContent}>
        {!surveyDone && (
          <View style={styles.answersGroupWrapper}>
            {questionsData[questionIndex] && groupedAnswers[questionsData[questionIndex].componentID] ? (
              groupedAnswers[questionsData[questionIndex].componentID].map((answer) => {
                let answerShort = answer.answerText;
                let answerDescription = answer.answerTextLong;
                let answerID = answer.answerID;
                let componentID = answer.componentID;

                return (
                  <Pressable
                    style={styles.answerButtonWrapper}
                    onPress={() => handleSelect(answerID, componentID)}
                    key={answerID}
                  >
                    <View
                      style={[
                        styles.answerButtonLabelWrapper,
                        activeAnswer === answerID && styles.activeAnswerButtonLabelWrapper,
                      ]}
                    >
                      <Text
                        style={[
                          styles.answerLabel,
                          textStyles.regular,
                          activeAnswer === answerID && styles.activeAnswerLabel,
                        ]}
                      >
                        {answerShort}
                      </Text>
                    </View>
                    <Text style={[textStyles.regular, styles.answerDesc]}>{answerDescription}</Text>
                  </Pressable>
                );
              })
            ) : (
              <Text style={{ ...textStyles.regular, fontSize: bodyFontSize }}>
                No answers available for this question.
              </Text>
            )}
          </View>
        )}
        {surveyDone && surveys.getLatestSurveyData && (
          <View style={styles.answersGroupWrapper}>
            <Image source={PRIMSLogo} style={{ height: 60, width: '100%', resizeMode: 'contain' }}></Image>
            <View style={{ flex: 1, justifyContent: 'center' }}>
              <Text style={{ ...textStyles.regular }}>Weekly Parkinson's Survey Complete</Text>
            </View>
            <Text style={{ ...textStyles.regular, flex: 1, textAlign: 'center' }}>
              This survey returned a rating of {surveys.getLatestSurveyData.surveyScore.toFixed(2)} {'\n'}Your results
              have been saved to your profile.
            </Text>

            <View style={styles.scaleWrapper}>
              <View style={styles.scaleGroup}>
                <View
                  style={{ ...styles.pointerGroup, left: `${(surveys.getLatestSurveyData.surveyScore / 4) * 100}%` }}
                >
                  <View style={styles.scoreBubble}>
                    <Text style={[textStyles.regular, styles.scoreBubbleText]}>
                      {surveys.getLatestSurveyData.surveyScore.toFixed(2)}
                    </Text>
                  </View>
                  <View style={styles.scalePointer}></View>
                </View>
                <View style={styles.scale1}></View>
                <View style={styles.scale2}></View>
                <View style={styles.scale3}></View>
                <View style={styles.scale4}></View>
              </View>
              <View style={styles.scaleLabelWrapper}>
                <Text style={[textStyles.regular, styles.scaleLabel]}>Normal</Text>
                <Text style={[textStyles.regular, styles.scaleLabel]}>Slight</Text>
                <Text style={[textStyles.regular, styles.scaleLabel]}>Mild</Text>
                <Text style={[textStyles.regular, styles.scaleLabel]}>Moderate</Text>
              </View>
            </View>
          </View>
        )}
        {surveyDone && !surveys.getLatestSurveyData && (
          <View style={{ alignItems: 'center', marginTop: topContainerHeight, flex: 1 }}>
            <Image source={gif} style={{ width: loaderWidth, height: loaderHeight }}></Image>
          </View>
        )}
        <View style={styles.nextButtonWrapper}>
          {toggleNextButton && (
            <Pressable
              onPress={handleNext}
              style={{
                ...styles.button,
                height: buttonHeight,
                width: buttonWidth,
              }}
            >
              <Text
                style={{
                  ...textStyles.regular,
                  fontSize: fontSize,
                  color: 'white',
                  marginTop: 'auto',
                  marginBottom: 'auto',
                }}
              >
                {nextButtonText}
              </Text>
            </Pressable>
          )}
        </View>
      </View>
      <View
        style={{ ...styles.navBarSpacer, height: isLargeScreen ? useResponsiveHeight(12) : useResponsiveHeight(10) }}
      ></View>
    </View>
  );
}
