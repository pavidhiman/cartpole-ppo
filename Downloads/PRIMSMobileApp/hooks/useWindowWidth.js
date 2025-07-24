import { useState, useEffect } from 'react';
import { Dimensions } from 'react-native';

export const useWindowWidth = () => {
  const [screenWidth, setScreenWidth] = useState(Dimensions.get('window').width);

  useEffect(() => {
    const handleResize = (newDimensions) => {
      setScreenWidth(newDimensions.window.width);
    };

    Dimensions.addEventListener('change', handleResize);

    return () => {};
  }, []);

  const isLargeScreen = screenWidth >= 992;

  return isLargeScreen;
};
