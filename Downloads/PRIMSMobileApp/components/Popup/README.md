# PRIMS Mobile PopUp Component

This component is a mobile popup menu that can contain a header, some text, adn up to two buttons within it.

The tinted view is configured with `Position: Absolute` and will cover the entire screen, disabling any buttons underneath it. the only interactions will be the button(s) within the popup.

## Properties

- **text (string)**
  - Provide a string of text to be displayed in the main body of your `Popup`
- **header (string)**
  - Provide a string of text to be displayed in the header of your `Popup`
- **buttonAmount (number)**
  - Provide a number (either 1 or 2) to specify how many buttons you want in your `Popup`. If two buttons are needed, they will be displayed in a row orientation. These buttons come from our `CustomButton` reusable component.
- **firstButtonText (string)**
  - Provide a string of text to be displayed in the first (left) button
- **secondButtonText (string)**
  - Provide a string of text to be displayed in the second (right) button
- **coloredButton (string)**
  - Specify which button should be colored. Set this to `right` or `left`
- **theme (string)**
  - Specify which color theme the buttons should match. These vary depending on which page the `Popup` is being used on.
    The options are: `general`, `appointment`, `survey` or `assessment`
- **isLargeScreen (boolean)**
  - Provide the `isLargeScreen` variable to the `Popup` component to help appropriately size the buttons. The value should come from the `useWindowWidth` hook
- **leftFunction (function)**
  - Provide a function to be called when the user presses the first (left) button.
- **rightFunction (function)**
  - Provide a function to be called when the user presses the second (right) button.

## Example Usage

```jsx
<Popup
  text={'Are you sure you want to log out of your account?'}
  header={'Log Out?'}
  buttonAmount={2}
  firstButtonText={'No'}
  secondButtonText={'Yes'}
  coloredButton={'right'}
  theme={'general'}
  isLargeScreen={isLargeScreen}
  leftFunction={handleLogOutPress}
></Popup>
```

### NOTES

- the default button color is `#F3F3F3`. if `coloredButton` is set to `right`, the left button will default to `#F3F3F3`
- The text color in the buttons will automatically change to white if the button is colored
