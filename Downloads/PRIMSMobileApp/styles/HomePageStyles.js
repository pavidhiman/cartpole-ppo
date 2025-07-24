import { StyleSheet } from 'react-native';

export const largeScreenStylesHome = StyleSheet.create({
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  box: {
    borderRadius: 10,
    marginLeft: 10,
    marginRight: 10,
    marginTop: 10,
  },
});

export const smallScreenStylesHome = StyleSheet.create({
  centerContent: {
    flex: 1,
    justifyContent: 'center',
  },
  box: {
    borderRadius: 10,
    marginTop: 10,
    marginBottom: 10,
  },
});

export const sharedStylesHome = StyleSheet.create({
  verticalLayout: {
    flexDirection: 'column',
  },
  horizontalLayout: {
    flexDirection: 'row',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  labelText: {
    color: 'white',
  },
  labelContainer: {
    width: '100%',
    justifyContent: 'center',
    borderRadius: 10,
    paddingLeft: 25,
  },
  bgImage: {
    width: 100,
    height: 100,
    position: 'absolute',
    bottom: 10,
    right: 20,
  },
  btnsContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    borderRadius: 10,
    backgroundColor: 'white',
  },
  buttonText: {
    color: 'black',
  },
});
