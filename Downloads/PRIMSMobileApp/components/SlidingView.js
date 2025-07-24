// components
import { StyleSheet, View, Text, Pressable, Dimensions } from 'react-native';
import { useResponsiveFontSize } from 'react-native-responsive-dimensions';
import DownArrow from './svg/DownArrow';

// styles
import { textStyles } from '../styles/textStyles';

export default function SlidingView({ children, heading, color, isOpen, onOpen }) {
  const headingFontSize = useResponsiveFontSize(2.5);
  return (
    <View style={styles.wrapper(color, isOpen)}>
      <View style={styles.content}>
        <Text style={{ ...styles.heading, fontSize: headingFontSize }}>{heading}</Text>
        {children}
      </View>

      <Text style={{ ...styles.heading, fontSize: headingFontSize }}>{heading}</Text>

      <Pressable onPress={onOpen}>
        <View style={styles.tab(color)}>
          <DownArrow />
        </View>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: (color, isOpen) => ({
    position: 'absolute',
    top: isOpen ? 0 : -Dimensions.get('window').height,
    backgroundColor: color,
    width: Dimensions.get('window').width,
    zIndex: 1,
  }),

  content: { height: Dimensions.get('window').height, overflow: 'hidden' },

  heading: {
    ...textStyles.bold,
    color: 'white',
    textAlign: 'center',
    height: 45,
    textAlignVertical: 'center',
  },

  tab: (color) => ({
    backgroundColor: color,
    height: 26,
    width: 198,
    position: 'absolute',
    left: '25%',
    justifyContent: 'center',
    alignItems: 'center',
    borderBottomLeftRadius: 12,
    borderBottomRightRadius: 12,
  }),
});
