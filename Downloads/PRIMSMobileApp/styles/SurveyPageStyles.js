import { StyleSheet } from 'react-native';

export const surveyPageStyles = StyleSheet.create({
  pageContainer: {
    width: '100%',
    height: '100%',
  },
  topContainer: {
    backgroundColor: '#3C6672',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  mainText: {
    color: 'white',
    marginTop: 15,
  },
  mainHeader: {
    color: 'white',
  },
  title: {
    marginBottom: 10,
    color: 'white',
  },

  topbarText: {
    // fontFamily: 'DM Sans',
    color: 'white',
  },

  mainContent: {
    flex: 4,
    // flexWrap: 'wrap',
    alignContent: 'center',
    justifyContent: 'center',
    paddingTop: 10,
    paddingBottom: 10,
  },

  answersGroupWrapper: {
    flex: 4,
    flexWrap: 'wrap',
    flexDirection: 'column',
    gap: 10,
    alignContent: 'center',
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 10,
    paddingBottom: 10,
    width: '80%',
    marginLeft: 'auto',
    marginRight: 'auto',
  },

  finalTextWrapper: {
    flex: 5,
  },

  answerButtonWrapper: {
    backgroundColor: '#f3f3f3',
    flex: 1,
    flexDirection: 'row',
    gap: 10,
    width: '100%',
    borderRadius: 10,
  },
  answerButtonLabelWrapper: {
    backgroundColor: '#D6D6D6',
    width: '25%',
    borderRadius: 10,
    justifyContent: 'center',
    paddingLeft: 5,
  },

  activeAnswerButtonLabelWrapper: {
    backgroundColor: '#3C6672',
  },
  answerLabel: {},

  activeAnswerLabel: {
    color: 'white',
  },

  answerDesc: {
    width: '70%',
    marginTop: 'auto',
    marginBottom: 'auto',
  },

  nextButtonWrapper: {
    flex: 1,
    justifyContent: 'center',
    alignContent: 'center',
  },

  scaleWrapper: {
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    height: 80,
  },

  scaleGroup: {
    flexDirection: 'row',
  },

  button: {
    backgroundColor: '#3C6672',
    justifyContent: 'center',
    alignContent: 'center',
    flexDirection: 'row',
    paddingTop: 'auto',
    paddingBottom: 'auto',
    borderRadius: 10,
    marginRight: 'auto',
    marginLeft: 'auto',
  },
  scale1: {
    height: 10,
    flex: 1,
    borderBottomLeftRadius: 10,
    borderTopLeftRadius: 10,
    backgroundColor: '#EFEFEF',
  },
  scale2: {
    height: 10,
    flex: 1,
    backgroundColor: '#DDDDDD',
  },
  scale3: {
    height: 10,
    flex: 1,
    backgroundColor: '#989898',
  },
  scale4: {
    height: 10,
    flex: 1,
    borderBottomRightRadius: 10,
    borderTopRightRadius: 10,
    backgroundColor: '#3D3D3D',
  },

  scaleLabelWrapper: {
    flexDirection: 'row',
    width: '120%',
  },

  scaleLabel: {
    flex: 1,
    textAlign: 'center',
    marginTop: 10,
  },

  pointerGroup: {
    height: 50,
    position: 'absolute',
    bottom: 0,
    zIndex: 4,
    backgroundColor: 'orange',
    flexDirection: 'column',
  },

  scalePointer: {
    backgroundColor: 'black',
    flex: 1,
    height: 30,
    width: 10,
    marginRight: 50,
    position: 'absolute',
    left: -5,
    bottom: -10,
    borderRadius: 10,
    // zIndex: 4,
  },

  scoreBubble: {
    backgroundColor: '#3C6672',
    position: 'absolute',
    left: -20,
    borderRadius: 20,
    top: -5,
  },

  scoreBubbleText: {
    color: 'white',
    marginTop: 6,
    marginBottom: 6,
    marginLeft: 10,
    marginRight: 10,
  },
});
