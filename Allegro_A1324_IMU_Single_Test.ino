#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

// ----------- Pins -----------
#define SDA_PIN A4
#define SCL_PIN A5

// ----------- Devices -----------
Adafruit_ADS1115 ads1; // 0x48
Adafruit_ADS1115 ads2; // 0x49
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

// ----------- Settings -----------
#define SEND_HEADER 1   // set to 0 if you don't want header

void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  // ----------- ADCs -----------
  if (!ads1.begin(0x48)) while (1);
  if (!ads2.begin(0x49)) while (1);

  ads1.setGain(GAIN_ONE);
  ads2.setGain(GAIN_ONE);

  // ----------- IMU -----------
  if (!bno.begin()) while (1);
  bno.setExtCrystalUse(true);

#if SEND_HEADER
  Serial.println("T,I,M,R,P,ax,ay,az,gx,gy,gz,ex,ey,ez");
#endif
}

void loop() {

  // ----------- Hall Sensors (Reordered for ML) -----------
  int16_t T = ads1.readADC_SingleEnded(0); // Thumb
  int16_t I = ads2.readADC_SingleEnded(0); // Index
  int16_t M = ads2.readADC_SingleEnded(1); // Middle
  int16_t R = ads2.readADC_SingleEnded(2); // Ring
  int16_t P = ads2.readADC_SingleEnded(3); // Pinky

  // ----------- IMU -----------
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> gyro  = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

  // ----------- CSV Output -----------

  Serial.print(T); Serial.print(",");
  Serial.print(I); Serial.print(",");
  Serial.print(M); Serial.print(",");
  Serial.print(R); Serial.print(",");
  Serial.print(P); Serial.print(",");

  Serial.print(accel.x()); Serial.print(",");
  Serial.print(accel.y()); Serial.print(",");
  Serial.print(accel.z()); Serial.print(",");

  Serial.print(gyro.x()); Serial.print(",");
  Serial.print(gyro.y()); Serial.print(",");
  Serial.print(gyro.z()); Serial.print(",");

  Serial.print(euler.x()); Serial.print(",");
  Serial.print(euler.y()); Serial.print(",");
  Serial.print(euler.z());

  Serial.println();

  delay(20); // ~50 Hz (good for ML)
}