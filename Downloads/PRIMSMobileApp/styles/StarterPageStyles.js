import { StyleSheet } from 'react-native';

export const starterPageStyles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    borderRadius: 10,
  },
  buttonText: {
    color: 'white',
  },
  patternImage: {
    zIndex: -1,
    position: 'absolute',
    right: 0,
    bottom: 0,
    width: 2604 / 4.5,
    height: 1592 / 4.5,
  },
});
