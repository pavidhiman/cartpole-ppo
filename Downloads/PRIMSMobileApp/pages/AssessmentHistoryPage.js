// components
import { Dimensions, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import HorizontalRule from '../components/HorizontalRule';
import SlidingView from '../components/SlidingView';

// PRIMS API
import { getOneYrAssessments } from '../slices/AssessmentSlice';

// hooks
import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';

// styles
import { textStyles } from '../styles/textStyles';

export default function AssessmentHistoryPage() {
  const assessments = useSelector((state) => state.assessments);
  const validUser = useSelector((state) => state.validUser);
  const dispatch = useDispatch();

  const [openSlider, setOpenSlider] = useState(true);
  const [assessmentSelected, setAssessmentSelected] = useState(false);
  const [currentAssessment, setCurrentAssessment] = useState({ name: null, date: null, score: null });

  const dummyData = [
    { name: '1. First Test', score: 0 },
    { name: '2. Second Test', score: 0 },
    { name: '3. Third Test', score: 0 },
    { name: '4. Fourth Test', score: 0 },
    { name: '5. Fifth Test', score: 0 },
    { name: '6. Sixth Test', score: 0 },
    { name: '7. Seventh Test', score: 0 },
    { name: '8. Eighth Test', score: 0 },
  ];

  const getAppointmentHistory = () => {
    dispatch(getOneYrAssessments({ patientID: validUser.data.patientID }));
  };

  const handleAssessmentSelect = (x) => {
    setAssessmentSelected(true);
    setCurrentAssessment({
      name: x.assessmentName,
      date: new Date(x.timeCompleted).toLocaleString('default', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      }),
      score: x.assessmentScore,
    });
    // TODO: add dispatch for getting assessment details
  };

  const handleAssessmentConfirm = () => {
    setOpenSlider(false);
    setAssessmentSelected(false);
  };

  useEffect(() => {
    getAppointmentHistory();
  }, []);

  return (
    <View style={assessmentHistoryStyles.wrapper}>
      <SlidingView
        heading="ASSESSMENT HISTORY"
        color="#84A258"
        isOpen={openSlider}
        onOpen={() => {
          setOpenSlider(true);
        }}
      >
        <View style={sliderStyles.assessmentsWrapper}>
          <ScrollView>
            {assessments.getOneYrAssessmentsData.map((x, i) => {
              return (
                <Pressable
                  key={i}
                  style={sliderStyles.assessment}
                  onPress={() => {
                    handleAssessmentSelect(x);
                  }}
                >
                  <View>
                    <Text style={{ ...textStyles.bold, fontSize: 16, color: 'white' }}>{x.assessmentName}</Text>
                    <Text style={{ ...textStyles.regular, fontSize: 16, color: 'white' }}>
                      {new Date(x.timeCompleted).toLocaleString('default', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </Text>
                  </View>
                  <Text style={{ ...textStyles.bold, fontSize: 20, color: 'white' }}>{x.assessmentScore}</Text>
                </Pressable>
              );
            })}
          </ScrollView>
        </View>

        <View style={sliderStyles.confirmWrapper}>
          {assessmentSelected && (
            <Pressable onPress={handleAssessmentConfirm}>
              <View style={sliderStyles.confirm}>
                <Text style={{ ...textStyles.regular, fontSize: 14 }}>Confirm</Text>
              </View>
            </Pressable>
          )}
        </View>
      </SlidingView>

      <View style={assessmentHistoryStyles.summary}>
        <View>
          <Text style={textStyles.bold}>{currentAssessment.name}</Text>
          <Text style={textStyles.regular}>{currentAssessment.date}</Text>
        </View>
        <Text style={{ ...textStyles.bold, fontSize: 28 }}>{currentAssessment.score}</Text>
      </View>

      <View style={assessmentHistoryStyles.legend}>
        <Text style={{ ...textStyles.regular, fontSize: 10, color: '#5E5E5E' }}>Test</Text>
        <Text style={{ ...textStyles.regular, fontSize: 10, color: '#5E5E5E' }}>Score</Text>
      </View>

      <ScrollView style={assessmentHistoryStyles.results}>
        {dummyData.map((x, i) => {
          return (
            <View key={i}>
              <View style={assessmentHistoryStyles.test}>
                <Text style={textStyles.regular}>{x.name}</Text>
                <Text style={textStyles.regular}>{x.score}</Text>
              </View>
              {i !== dummyData.length - 1 && <HorizontalRule color="#E8E8E8" />}
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}

const sliderStyles = StyleSheet.create({
  assessmentsWrapper: {
    height: Dimensions.get('window').height * (2 / 3),
    paddingTop: 57,
    paddingRight: 13,
    paddingLeft: 13,
  },

  assessment: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 18,
    borderWidth: 1,
    borderRadius: 12,
    borderColor: '#4D5640',
    marginBottom: 9,
  },

  confirmWrapper: { alignItems: 'center', marginTop: 18 },

  confirm: {
    height: 30,
    width: 185,
    borderRadius: 12,
    backgroundColor: '#D9D9D9',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

const assessmentHistoryStyles = StyleSheet.create({
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

  test: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', height: 90 },
});
