import { Pressable, View, StyleSheet, Text } from 'react-native';
import { textStyles } from '../styles/textStyles';

export default function Dropdown({ isOpen, options, onSelect, color }) {
  return (
    <View>
      {isOpen && (
        <View style={styles.menu(color)}>
          {options.map((option, index) => {
            return (
              <View key={index}>
                <Pressable
                  onPress={() => {
                    // calling provided onSelect function with the option the user selected
                    onSelect(option);
                  }}
                >
                  <Text style={styles.text}>{option}</Text>
                </Pressable>
              </View>
            );
          })}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  menu: (color) => {
    return { borderRadius: 12, borderColor: color, borderWidth: 1, paddingLeft: 10, paddingRight: 10 };
  },

  text: {
    ...textStyles.regular,
    paddingTop: 10,
    paddingBottom: 10,
  },
});
