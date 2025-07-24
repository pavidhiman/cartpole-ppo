import { StyleSheet } from 'react-native';

export const settingsPageStyles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
  },
  topContainer: {
    backgroundColor: '#95B0AA',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  mainHeader: {
    color: 'white',
  },
  label: {
    marginBottom: 5,
    marginTop: 60,
  },
  mainText: {
    color: 'white',
    marginTop: 15,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  button: {
    borderRadius: 10,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 30,
    paddingLeft: 30,
    paddingTop: 15,
    paddingBottom: 15,
    width: '100%',
  },
  settingsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderRadius: 5,
  },
  settingsButtonText: {
    marginLeft: 10,
  },
});
