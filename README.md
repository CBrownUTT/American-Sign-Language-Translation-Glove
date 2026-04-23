# The Wearable Glove for American Sign Langauge (ASL) Gesture Recognition  
**The Wearable Glove for American Sign Langauge (ASL) Gesture Recognition** is a prototype developed by **Jalen Bell**, **Chase Brown**, and **Jared Carrig**.
The system is able to predict all of the ASL alphabet and numbers one through ten. 
The glove features dynamic gesture recognition, real-time functionality, portability, predictions over BLE, and implements three major sensor subsystems.
Development began on January 13th and ended April 17th. 
### Video Performance
If you are interested in a video of the functioning prototype, please click this [LINK!](https://www.youtube.com/watch?v=rQbK6q550VY)
### More Information
If you would like to know more about the development of the project or other specifics, please find the **'Final Design Report'** and **'Final Design Presentation'** within the **'Docs'** folder.
As for testing the prototype and it's code yourself, go to the **'modelWrapperBLE'** folder and upload the **'modelWrapperBLE.ino'** file with a baud rate of 115200 to your Arduino ESP32 Nano.
This arduino file containts the model that makes predictions based on sensory data and outputs the top three predictions through serial.
For BLE output go to the **'BLE Output Program'** folder, which located wihin the same directory as the .ino file, and run **'ModelDisplay.py'**.
