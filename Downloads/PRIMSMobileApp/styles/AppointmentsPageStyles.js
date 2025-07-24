import { StyleSheet } from 'react-native';

export const apptsStyles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
  },
  contentContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  topContainer: {
    backgroundColor: '#DEBE5B',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  mainText: {
    marginTop: 15,
  },
  border: {
    backgroundColor: '#F3F3F3',
    borderRadius: 25,
    borderWidth: 1,
    borderColor: '#D9D9D9',
    marginTop: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  button: {
    borderRadius: 15,
  },
  pickerLabel: {
    marginBottom: 5,
  },
  contentContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});
