#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "feature_builder.h"
#include "asl_rf_model.h"

using namespace ASLModel;

#define SDA_PIN A4
#define SCL_PIN A5

#define PIN_CONTACT_P   A0
#define PIN_CONTACT_I   A1
#define PIN_CONTACT_M   A2
#define PIN_CONTACT_UM  A3

#define BLE_DEVICE_NAME   "ASL_Glove"
#define SERVICE_UUID      "12345678-1234-1234-1234-1234567890ab"
#define RESULT_CHAR_UUID  "12345678-1234-1234-1234-1234567890ac"

// Trouble with F,Q,U,9

Adafruit_ADS1115 ads1;
Adafruit_ADS1115 ads2;
Adafruit_BNO055 bno28(55, 0x28);
Adafruit_BNO055 bno29(56, 0x29);
Adafruit_BNO055* bno = nullptr;

static constexpr int FRAME_COUNT = 15;
static constexpr int SAMPLE_DELAY_MS = 20;

BLEServer* bleServer = nullptr;
BLECharacteristic* resultChar = nullptr;
bool bleClientConnected = false;

Frame g_frames[FRAME_COUNT];
float g_features[FEATURE_COUNT];
float g_proba[kClassCount];

float g_ax = 0.0f, g_ay = 0.0f, g_az = 0.0f;
float g_gx = 0.0f, g_gy = 0.0f, g_gz = 0.0f;
float g_ex = 0.0f, g_ey = 0.0f, g_ez = 0.0f;

class GloveServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    bleClientConnected = true;
    Serial.println("BLE client connected");
  }

  void onDisconnect(BLEServer* pServer) override {
    bleClientConnected = false;
    Serial.println("BLE client disconnected");
    BLEDevice::startAdvertising();
  }
};

void fatalError(const char* msg) {
  Serial.println(msg);
  while (1) {
    delay(100);
  }
}

float readHallThumb()  { return (float)ads1.readADC_SingleEnded(0); }
float readHallIndex()  { return (float)ads2.readADC_SingleEnded(0); }
float readHallMiddle() { return (float)ads2.readADC_SingleEnded(1); }
float readHallRing()   { return (float)ads2.readADC_SingleEnded(2); }
float readHallPinky()  { return (float)ads2.readADC_SingleEnded(3); }

float readContactP()  { return (float)digitalRead(PIN_CONTACT_P); }
float readContactI()  { return (float)digitalRead(PIN_CONTACT_I); }
float readContactM()  { return (float)digitalRead(PIN_CONTACT_M); }
float readContactUM() { return (float)digitalRead(PIN_CONTACT_UM); }

void updateIMU() {
  imu::Vector<3> accel = bno->getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> gyro  = bno->getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> euler = bno->getVector(Adafruit_BNO055::VECTOR_EULER);

  g_ax = accel.x();
  g_ay = accel.y();
  g_az = accel.z();

  g_gx = gyro.x();
  g_gy = gyro.y();
  g_gz = gyro.z();

  g_ex = euler.x();
  g_ey = euler.y();
  g_ez = euler.z();
}

bool initHardware() {
  pinMode(PIN_CONTACT_P, INPUT_PULLDOWN);
  pinMode(PIN_CONTACT_I, INPUT_PULLDOWN);
  pinMode(PIN_CONTACT_M, INPUT_PULLDOWN);
  pinMode(PIN_CONTACT_UM, INPUT_PULLDOWN);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!ads1.begin(0x48, &Wire)) return false;
  if (!ads2.begin(0x49, &Wire)) return false;

  ads1.setGain(GAIN_ONE);
  ads2.setGain(GAIN_ONE);

  if (bno28.begin()) {
    bno = &bno28;
  } else if (bno29.begin()) {
    bno = &bno29;
  } else {
    return false;
  }

  bno->setExtCrystalUse(true);
  delay(1000);
  return true;
}

void initBLE() {
  BLEDevice::init(BLE_DEVICE_NAME);

  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new GloveServerCallbacks());

  BLEService* service = bleServer->createService(SERVICE_UUID);

  resultChar = service->createCharacteristic(
    RESULT_CHAR_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );

  resultChar->addDescriptor(new BLE2902());
  resultChar->setValue("label=none,idx=-1,conf=0.00");

  service->start();

  BLEAdvertising* advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->start();

  Serial.println("BLE advertising started");
}

void fillFrame(Frame &frame) {
  updateIMU();

  frame.sensor[0]  = readHallThumb();
  frame.sensor[1]  = readHallIndex();
  frame.sensor[2]  = readHallMiddle();
  frame.sensor[3]  = readHallRing();
  frame.sensor[4]  = readHallPinky();

  frame.sensor[5]  = g_ax;
  frame.sensor[6]  = g_ay;
  frame.sensor[7]  = g_az;

  frame.sensor[8]  = g_gx;
  frame.sensor[9]  = g_gy;
  frame.sensor[10] = g_gz;

  frame.sensor[11] = g_ex;
  frame.sensor[12] = g_ey;
  frame.sensor[13] = g_ez;

  frame.sensor[14] = readContactP();
  frame.sensor[15] = readContactI();
  frame.sensor[16] = readContactM();
  frame.sensor[17] = readContactUM();

  frame.timestamp_ms = millis();
}

bool captureAndPredict(int frameCount, int& classIdx, const char*& classLabel) {
  if (frameCount < 2 || frameCount > FRAME_COUNT) return false;

  for (int i = 0; i < frameCount; ++i) {
    fillFrame(g_frames[i]);
    delay(SAMPLE_DELAY_MS);
  }

  bool ok = build_feature_vector(g_frames, frameCount, g_features);
  if (!ok) return false;

  predict_proba(g_features, g_proba);
  classIdx = predict(g_features);
  if (classIdx < 0 || classIdx >= kClassCount) return false;

  classLabel = kClassLabels[classIdx];
  return true;
}

void printTop3() {
  int best1 = -1, best2 = -1, best3 = -1;

  for (int i = 0; i < kClassCount; ++i) {
    if (best1 < 0 || g_proba[i] > g_proba[best1]) {
      best3 = best2;
      best2 = best1;
      best1 = i;
    } else if (best2 < 0 || g_proba[i] > g_proba[best2]) {
      best3 = best2;
      best2 = i;
    } else if (best3 < 0 || g_proba[i] > g_proba[best3]) {
      best3 = i;
    }
  }

  Serial.println("Top 3 predictions:");
  if (best1 >= 0) {
    Serial.print("  1) ");
    Serial.print(kClassLabels[best1]);
    Serial.print(" = ");
    Serial.println(g_proba[best1], 6);
  }
  if (best2 >= 0) {
    Serial.print("  2) ");
    Serial.print(kClassLabels[best2]);
    Serial.print(" = ");
    Serial.println(g_proba[best2], 6);
  }
  if (best3 >= 0) {
    Serial.print("  3) ");
    Serial.print(kClassLabels[best3]);
    Serial.print(" = ");
    Serial.println(g_proba[best3], 6);
  }
}

void sendPredictionOverBLE(const char* classLabel, int classIdx, float confidence) {
  if (!bleClientConnected || resultChar == nullptr) {
    return;
  }

  char payload[80];
  snprintf(payload, sizeof(payload), "label=%s,idx=%d,conf=%.2f", classLabel, classIdx, confidence);

  resultChar->setValue((uint8_t*)payload, strlen(payload));
  resultChar->notify();

  Serial.print("BLE sent: ");
  Serial.println(payload);
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  if (!initHardware()) {
    fatalError("Hardware init failed");
  }

  initBLE();
  Serial.println("System ready");
}

void loop() {
  int classIdx = -1;
  const char* classLabel = "";

  bool ok = captureAndPredict(FRAME_COUNT, classIdx, classLabel);

  if (!ok) {
    Serial.println("Capture/predict failed");
    delay(500);
    return;
  }

  Serial.print("Predicted index: ");
  Serial.println(classIdx);
  Serial.print("Predicted label: ");
  Serial.println(classLabel);
  Serial.print("Confidence: ");
  Serial.println(g_proba[classIdx], 6);

  sendPredictionOverBLE(classLabel, classIdx, g_proba[classIdx]);
  printTop3();
  delay(1000);
}
