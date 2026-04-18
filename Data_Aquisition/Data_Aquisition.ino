#include <Wire.h> // Lets the board talk over I2C
#include <Adafruit_ADS1X15.h> // Library for the ADS1115 ADC boards
#include <Adafruit_BNO055.h> // Library for the BNO055 IMU
#include <Adafruit_Sensor.h> // Required helper library for Adafruit sensors

// ----------- Digital Contact Pins -----------
#define PALM          17 // Pin for the palm contact sensor
#define INDEX         18 // Pin for the index finger contact sensor
#define MIDDLE        19 // Pin for the middle finger contact sensor
#define UNDER_MIDDLE  20 // Pin for the under-middle contact sensor

// ----------- Debounce -----------
#define DEBOUNCE_MS 20 // Debounce time in milliseconds

// ----------- Devices -----------
Adafruit_ADS1115 ads1; // First ADS1115 ADC at I2C address 0x48
Adafruit_ADS1115 ads2; // Second ADS1115 ADC at I2C address 0x49
Adafruit_BNO055 bno28(55, 0x28); // BNO055 IMU object for address 0x28
Adafruit_BNO055 bno29(56, 0x29); // BNO055 IMU object for address 0x29
Adafruit_BNO055* bno = nullptr; // Pointer that will point to whichever IMU is found

void fatalError(const char* msg) { // Stops the program if a critical device is missing
  Serial.println(msg); // Print the error message to the Serial Monitor
  while (1) { // Stay here forever so the program does not continue
    delay(100); // Small delay so the loop does not run too fast
  }
}

void setup() {
  Serial.begin(115200); // Start Serial communication
  delay(2000); // Give time for Serial to fully start

  // Digital contacts
  pinMode(PALM, INPUT_PULLDOWN); // Set the palm contact pin as input with pulldown
  pinMode(INDEX, INPUT_PULLDOWN); // Set the index contact pin as input with pulldown
  pinMode(MIDDLE, INPUT_PULLDOWN); // Set the middle contact pin as input with pulldown
  pinMode(UNDER_MIDDLE, INPUT_PULLDOWN); // Set the under-middle contact pin as input with pulldown

  // I2C
  Wire.begin(); // Start I2C communication
  Wire.setClock(400000); // Set I2C speed to 400 kHz

  if (!ads1.begin(0x48, &Wire)) { // Try to start the first ADS1115
    fatalError("ERROR: ADS1115 at 0x48 not found"); // Stop if it is not found
  }

  if (!ads2.begin(0x49, &Wire)) { // Try to start the second ADS1115
    fatalError("ERROR: ADS1115 at 0x49 not found"); // Stop if it is not found
  }

  ads1.setGain(GAIN_ONE); // Set gain for the first ADS1115
  ads2.setGain(GAIN_ONE); // Set gain for the second ADS1115

  if (bno28.begin()) { // Try to start the BNO055 at address 0x28
    bno = &bno28; // Use this IMU if found
  } else if (bno29.begin()) { // Otherwise try the BNO055 at address 0x29
    bno = &bno29; // Use this IMU if found
  } else {
    fatalError("ERROR: BNO055 not found at 0x28 or 0x29"); // Stop if no IMU is found
  }

  bno->setExtCrystalUse(true); // Use the external crystal for better IMU accuracy
  delay(1000); // Give the IMU time to settle

  Serial.println("hall_thumb,hall_index,hall_middle,hall_ring,hall_pinky,imu_ax,imu_ay,imu_az,imu_gx,imu_gy,imu_gz,imu_ex,imu_ey,imu_ez,contact_p,contact_i,contact_m,contact_um"); // Print the CSV header line
}

void loop(){
  
  

  // Hall sensors
  int16_t hall_thumb  = ads1.readADC_SingleEnded(0); // Read thumb hall sensor from ads1 channel 0
  int16_t hall_index  = ads2.readADC_SingleEnded(0); // Read index hall sensor from ads2 channel 0
  int16_t hall_middle = ads2.readADC_SingleEnded(1); // Read middle hall sensor from ads2 channel 1
  int16_t hall_ring   = ads2.readADC_SingleEnded(2); // Read ring hall sensor from ads2 channel 2
  int16_t hall_pinky  = ads2.readADC_SingleEnded(3); // Read pinky hall sensor from ads2 channel 3

  // IMU
  imu::Vector<3> accel = bno->getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // Read accelerometer values
  imu::Vector<3> gyro  = bno->getVector(Adafruit_BNO055::VECTOR_GYROSCOPE); // Read gyroscope values
  imu::Vector<3> euler = bno->getVector(Adafruit_BNO055::VECTOR_EULER); // Read Euler angle values

  // Contacts
  int contact_p  = digitalRead(PALM); // Read the palm contact sensor
  int contact_i  = digitalRead(INDEX); // Read the index contact sensor
  int contact_m  = digitalRead(MIDDLE); // Read the middle contact sensor
  int contact_um = digitalRead(UNDER_MIDDLE); // Read the under-middle contact sensor

  // CSV output
  Serial.print(hall_thumb);  Serial.print(","); // Print thumb hall value
  Serial.print(hall_index);  Serial.print(","); // Print index hall value
  Serial.print(hall_middle); Serial.print(","); // Print middle hall value
  Serial.print(hall_ring);   Serial.print(","); // Print ring hall value
  Serial.print(hall_pinky);  Serial.print(","); // Print pinky hall value

  Serial.print(accel.x(), 6); Serial.print(","); // Print accelerometer X value
  Serial.print(accel.y(), 6); Serial.print(","); // Print accelerometer Y value
  Serial.print(accel.z(), 6); Serial.print(","); // Print accelerometer Z value

  Serial.print(gyro.x(), 6); Serial.print(","); // Print gyroscope X value
  Serial.print(gyro.y(), 6); Serial.print(","); // Print gyroscope Y value
  Serial.print(gyro.z(), 6); Serial.print(","); // Print gyroscope Z value

  Serial.print(euler.x(), 6); Serial.print(","); // Print Euler X value
  Serial.print(euler.y(), 6); Serial.print(","); // Print Euler Y value
  Serial.print(euler.z(), 6); Serial.print(","); // Print Euler Z value

  Serial.print(contact_p); Serial.print(","); // Print palm contact value
  Serial.print(contact_i); Serial.print(","); // Print index contact value
  Serial.print(contact_m); Serial.print(","); // Print middle contact value
  Serial.println(contact_um); // Print under-middle contact value and move to next line

  delay(20); // Wait 20 ms before the next reading, about 50 Hz
}