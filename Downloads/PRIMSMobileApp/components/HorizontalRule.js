import { View } from 'react-native';

export default function HorizontalRule({ color }) {
  return (
    <View
      style={{
        borderBottomColor: color,
        borderBottomWidth: 1,
      }}
    />
  );
}
