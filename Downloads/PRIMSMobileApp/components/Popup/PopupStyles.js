import { StyleSheet } from 'react-native';
// import { useFonts, DMSans_400Regular } from '@expo-google-fonts/dm-sans';

export const popupStyles = StyleSheet.create({
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

  tint: {
    position: 'absolute',
    zIndex: 4,
    height: '100%',
    width: '100%',
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  popupFrame: {
    opacity: 1,
    backgroundColor: 'white',
    borderRadius: 10,
    elevation: 15,
    shadowColor: 'black',
    shadowRadius: 15,
    shadowOpacity: 0.3,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 30,
  },
  buttonWrapper: {
    flexDirection: 'row',
    gap: 20,
    marginBottom: 10,
  },

  popupText: {
    // fontFamily: 'DMSans_400Regular',
    marginBottom: 30,
  },

  popupHeader: {
    // fontFamily: 'DMSans_400Regular',
    marginBottom: 30,
  },
});
