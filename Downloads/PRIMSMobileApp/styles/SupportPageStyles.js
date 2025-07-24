import { StyleSheet } from 'react-native';

export const supportPageStyles = StyleSheet.create({
  testContainer: {
    width: '100%',
    height: '100%',
  },
  topContainer: {
    backgroundColor: '#95B0AA',
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
  mainContent: {
    height: '70%',
    flex: 1,
    flexWrap: 'wrap',
    alignContent: 'center',
    justifyContent: 'center',
  },
  buttonWrapper: {
    height: '80%',
    width: '80%',
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 70,
  },

  logoutWrapper: {
    width: '85%',
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

  primsLogo: {
    height: '10%',
    // width: 200,
  },
});
