import { StyleSheet } from 'react-native';

export const surveySelectPageStyles = StyleSheet.create({
  testContainer: {
    width: '100%',
    height: '100%',
  },
  topContainer: {
    backgroundColor: '#3C6672',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  title: {
    marginBottom: 10,
    color: 'white',
  },
  mainText: {
    color: 'white',
    marginTop: 15,
  },
  mainHeader: {
    color: 'white',
  },
  mainContent: {
    height: '50%',
    flex: 1,
    flexWrap: 'wrap',
    alignContent: 'center',
    justifyContent: 'center',
    marginTop: 100,
  },
  buttonWrapper: {
    height: '80%',
    width: '80%',
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
  },

  dropDownWrapper: {
    width: '80%',
    marginBottom: 40,
  },

  logoutButton: {
    width: '100%',
    height: 60,
    backgroundColor: '#F3F3F3',
    borderRadius: 10,
  },

  button: {
    backgroundColor: '#F3F3F3',
    borderRadius: 10,
    // fontFamily: 'DMSans_400Regular',
    justifyContent: 'flex-start',
    alignContent: 'flex-start',
    flexDirection: 'row',
  },

  buttonText: {
    // fontFamily: 'DMSans_400Regular',
  },

  topbarText: {
    // fontFamily: 'DM Sans',
    color: 'white',
  },
});
