#include <Wire.h> // Lets the board use I2C
#include <Adafruit_ADS1X15.h> // Library for ADS1115 ADC boards
#include <Adafruit_BNO055.h> // Library for the BNO055 IMU
#include <Adafruit_Sensor.h> // Helper sensor library
#include <math.h> // Needed for sqrt
#include <string.h> // Needed for memset
#include "rf_model.h" // Your exported random forest model

// ---------------- Contact Pins ----------------
#define PALM          17 // Palm contact sensor pin
#define INDEX         18 // Index contact sensor pin
#define MIDDLE        19 // Middle contact sensor pin
#define UNDER_MIDDLE  20 // Under-middle contact sensor pin

// ---------------- Timing ----------------
#define SAMPLE_DELAY_MS 20 // 20 ms between samples gives about 50 Hz
#define MAX_FRAMES 25 // Number of frames to store for one gesture window

// ---------------- Devices ----------------
Adafruit_ADS1115 ads1; // First ADS1115 at 0x48
Adafruit_ADS1115 ads2; // Second ADS1115 at 0x49
Adafruit_BNO055 bno28(55, 0x28); // BNO055 option at address 0x28
Adafruit_BNO055 bno29(56, 0x29); // BNO055 option at address 0x29
Adafruit_BNO055* bno = nullptr; // Pointer to whichever BNO055 is found

// ---------------- Raw Frame Struct ----------------
// This holds one raw sensor sample from the glove
struct RawFrame {
  unsigned long timestamp_ms; // Time this sample was taken

  float hall_thumb; // Thumb hall sensor
  float hall_index; // Index hall sensor
  float hall_middle; // Middle hall sensor
  float hall_ring; // Ring hall sensor
  float hall_pinky; // Pinky hall sensor

  float imu_ax; // Accelerometer X
  float imu_ay; // Accelerometer Y
  float imu_az; // Accelerometer Z

  float imu_gx; // Gyroscope X
  float imu_gy; // Gyroscope Y
  float imu_gz; // Gyroscope Z

  float imu_mx; // Magnetometer X
  float imu_my; // Magnetometer Y
  float imu_mz; // Magnetometer Z

  float contact_p; // Palm contact
  float contact_i; // Index contact
  float contact_m; // Middle contact
  float contact_um; // Under-middle contact
};

// ---------------- Frame Buffer ----------------
// This stores one full gesture sequence before feature extraction
RawFrame g_frames[MAX_FRAMES]; // Array of raw samples
int g_frameCount = 0; // Current number of stored frames

// ---------------- Feature Map Enum ----------------
// These indexes define exactly where each feature goes in the 284-element model input
// The order must match RFModel::FEATURE_NAMES exactly
enum FeatureIndex {
  IDX_duration_s = 0,
  IDX_num_frames = 1,

  IDX_hall_thumb_first = 2,
  IDX_hall_thumb_last,
  IDX_hall_thumb_mean,
  IDX_hall_thumb_min,
  IDX_hall_thumb_max,
  IDX_hall_thumb_std,
  IDX_hall_thumb_range,
  IDX_hall_thumb_delta,
  IDX_hall_thumb_abs_change_sum,
  IDX_hall_thumb_slope,
  IDX_hall_thumb_p0,
  IDX_hall_thumb_p25,
  IDX_hall_thumb_p50,
  IDX_hall_thumb_p75,
  IDX_hall_thumb_p100,

  IDX_hall_index_first,
  IDX_hall_index_last,
  IDX_hall_index_mean,
  IDX_hall_index_min,
  IDX_hall_index_max,
  IDX_hall_index_std,
  IDX_hall_index_range,
  IDX_hall_index_delta,
  IDX_hall_index_abs_change_sum,
  IDX_hall_index_slope,
  IDX_hall_index_p0,
  IDX_hall_index_p25,
  IDX_hall_index_p50,
  IDX_hall_index_p75,
  IDX_hall_index_p100,

  IDX_hall_middle_first,
  IDX_hall_middle_last,
  IDX_hall_middle_mean,
  IDX_hall_middle_min,
  IDX_hall_middle_max,
  IDX_hall_middle_std,
  IDX_hall_middle_range,
  IDX_hall_middle_delta,
  IDX_hall_middle_abs_change_sum,
  IDX_hall_middle_slope,
  IDX_hall_middle_p0,
  IDX_hall_middle_p25,
  IDX_hall_middle_p50,
  IDX_hall_middle_p75,
  IDX_hall_middle_p100,

  IDX_hall_ring_first,
  IDX_hall_ring_last,
  IDX_hall_ring_mean,
  IDX_hall_ring_min,
  IDX_hall_ring_max,
  IDX_hall_ring_std,
  IDX_hall_ring_range,
  IDX_hall_ring_delta,
  IDX_hall_ring_abs_change_sum,
  IDX_hall_ring_slope,
  IDX_hall_ring_p0,
  IDX_hall_ring_p25,
  IDX_hall_ring_p50,
  IDX_hall_ring_p75,
  IDX_hall_ring_p100,

  IDX_hall_pinky_first,
  IDX_hall_pinky_last,
  IDX_hall_pinky_mean,
  IDX_hall_pinky_min,
  IDX_hall_pinky_max,
  IDX_hall_pinky_std,
  IDX_hall_pinky_range,
  IDX_hall_pinky_delta,
  IDX_hall_pinky_abs_change_sum,
  IDX_hall_pinky_slope,
  IDX_hall_pinky_p0,
  IDX_hall_pinky_p25,
  IDX_hall_pinky_p50,
  IDX_hall_pinky_p75,
  IDX_hall_pinky_p100,

  IDX_imu_ax_first,
  IDX_imu_ax_last,
  IDX_imu_ax_mean,
  IDX_imu_ax_min,
  IDX_imu_ax_max,
  IDX_imu_ax_std,
  IDX_imu_ax_range,
  IDX_imu_ax_delta,
  IDX_imu_ax_abs_change_sum,
  IDX_imu_ax_slope,
  IDX_imu_ax_p0,
  IDX_imu_ax_p25,
  IDX_imu_ax_p50,
  IDX_imu_ax_p75,
  IDX_imu_ax_p100,

  IDX_imu_ay_first,
  IDX_imu_ay_last,
  IDX_imu_ay_mean,
  IDX_imu_ay_min,
  IDX_imu_ay_max,
  IDX_imu_ay_std,
  IDX_imu_ay_range,
  IDX_imu_ay_delta,
  IDX_imu_ay_abs_change_sum,
  IDX_imu_ay_slope,
  IDX_imu_ay_p0,
  IDX_imu_ay_p25,
  IDX_imu_ay_p50,
  IDX_imu_ay_p75,
  IDX_imu_ay_p100,

  IDX_imu_az_first,
  IDX_imu_az_last,
  IDX_imu_az_mean,
  IDX_imu_az_min,
  IDX_imu_az_max,
  IDX_imu_az_std,
  IDX_imu_az_range,
  IDX_imu_az_delta,
  IDX_imu_az_abs_change_sum,
  IDX_imu_az_slope,
  IDX_imu_az_p0,
  IDX_imu_az_p25,
  IDX_imu_az_p50,
  IDX_imu_az_p75,
  IDX_imu_az_p100,

  IDX_imu_gx_first,
  IDX_imu_gx_last,
  IDX_imu_gx_mean,
  IDX_imu_gx_min,
  IDX_imu_gx_max,
  IDX_imu_gx_std,
  IDX_imu_gx_range,
  IDX_imu_gx_delta,
  IDX_imu_gx_abs_change_sum,
  IDX_imu_gx_slope,
  IDX_imu_gx_p0,
  IDX_imu_gx_p25,
  IDX_imu_gx_p50,
  IDX_imu_gx_p75,
  IDX_imu_gx_p100,

  IDX_imu_gy_first,
  IDX_imu_gy_last,
  IDX_imu_gy_mean,
  IDX_imu_gy_min,
  IDX_imu_gy_max,
  IDX_imu_gy_std,
  IDX_imu_gy_range,
  IDX_imu_gy_delta,
  IDX_imu_gy_abs_change_sum,
  IDX_imu_gy_slope,
  IDX_imu_gy_p0,
  IDX_imu_gy_p25,
  IDX_imu_gy_p50,
  IDX_imu_gy_p75,
  IDX_imu_gy_p100,

  IDX_imu_gz_first,
  IDX_imu_gz_last,
  IDX_imu_gz_mean,
  IDX_imu_gz_min,
  IDX_imu_gz_max,
  IDX_imu_gz_std,
  IDX_imu_gz_range,
  IDX_imu_gz_delta,
  IDX_imu_gz_abs_change_sum,
  IDX_imu_gz_slope,
  IDX_imu_gz_p0,
  IDX_imu_gz_p25,
  IDX_imu_gz_p50,
  IDX_imu_gz_p75,
  IDX_imu_gz_p100,

  IDX_imu_mx_first,
  IDX_imu_mx_last,
  IDX_imu_mx_mean,
  IDX_imu_mx_min,
  IDX_imu_mx_max,
  IDX_imu_mx_std,
  IDX_imu_mx_range,
  IDX_imu_mx_delta,
  IDX_imu_mx_abs_change_sum,
  IDX_imu_mx_slope,
  IDX_imu_mx_p0,
  IDX_imu_mx_p25,
  IDX_imu_mx_p50,
  IDX_imu_mx_p75,
  IDX_imu_mx_p100,

  IDX_imu_my_first,
  IDX_imu_my_last,
  IDX_imu_my_mean,
  IDX_imu_my_min,
  IDX_imu_my_max,
  IDX_imu_my_std,
  IDX_imu_my_range,
  IDX_imu_my_delta,
  IDX_imu_my_abs_change_sum,
  IDX_imu_my_slope,
  IDX_imu_my_p0,
  IDX_imu_my_p25,
  IDX_imu_my_p50,
  IDX_imu_my_p75,
  IDX_imu_my_p100,

  IDX_imu_mz_first,
  IDX_imu_mz_last,
  IDX_imu_mz_mean,
  IDX_imu_mz_min,
  IDX_imu_mz_max,
  IDX_imu_mz_std,
  IDX_imu_mz_range,
  IDX_imu_mz_delta,
  IDX_imu_mz_abs_change_sum,
  IDX_imu_mz_slope,
  IDX_imu_mz_p0,
  IDX_imu_mz_p25,
  IDX_imu_mz_p50,
  IDX_imu_mz_p75,
  IDX_imu_mz_p100,

  IDX_contact_p_first,
  IDX_contact_p_last,
  IDX_contact_p_mean,
  IDX_contact_p_min,
  IDX_contact_p_max,
  IDX_contact_p_std,
  IDX_contact_p_range,
  IDX_contact_p_delta,
  IDX_contact_p_abs_change_sum,
  IDX_contact_p_slope,
  IDX_contact_p_p0,
  IDX_contact_p_p25,
  IDX_contact_p_p50,
  IDX_contact_p_p75,
  IDX_contact_p_p100,

  IDX_contact_i_first,
  IDX_contact_i_last,
  IDX_contact_i_mean,
  IDX_contact_i_min,
  IDX_contact_i_max,
  IDX_contact_i_std,
  IDX_contact_i_range,
  IDX_contact_i_delta,
  IDX_contact_i_abs_change_sum,
  IDX_contact_i_slope,
  IDX_contact_i_p0,
  IDX_contact_i_p25,
  IDX_contact_i_p50,
  IDX_contact_i_p75,
  IDX_contact_i_p100,

  IDX_contact_m_first,
  IDX_contact_m_last,
  IDX_contact_m_mean,
  IDX_contact_m_min,
  IDX_contact_m_max,
  IDX_contact_m_std,
  IDX_contact_m_range,
  IDX_contact_m_delta,
  IDX_contact_m_abs_change_sum,
  IDX_contact_m_slope,
  IDX_contact_m_p0,
  IDX_contact_m_p25,
  IDX_contact_m_p50,
  IDX_contact_m_p75,
  IDX_contact_m_p100,

  IDX_contact_um_first,
  IDX_contact_um_last,
  IDX_contact_um_mean,
  IDX_contact_um_min,
  IDX_contact_um_max,
  IDX_contact_um_std,
  IDX_contact_um_range,
  IDX_contact_um_delta,
  IDX_contact_um_abs_change_sum,
  IDX_contact_um_slope,
  IDX_contact_um_p0,
  IDX_contact_um_p25,
  IDX_contact_um_p50,
  IDX_contact_um_p75,
  IDX_contact_um_p100,

  IDX_accel_mag_mean,
  IDX_accel_mag_max,
  IDX_accel_mag_std,
  IDX_accel_mag_abs_change_sum,
  IDX_gyro_mag_mean,
  IDX_gyro_mag_max,
  IDX_gyro_mag_std,
  IDX_gyro_mag_abs_change_sum,
  IDX_mag_mag_mean,
  IDX_mag_mag_max,
  IDX_mag_mag_std,
  IDX_mag_mag_abs_change_sum
};

// ---------------- Error Handler ----------------
void fatalError(const char* msg) {
  Serial.println(msg); // Print the error
  while (1) { // Stop forever
    delay(100); // Small delay while stopped
  }
}

// ---------------- Small Math Helpers ----------------
float safeDivide(float a, float b) {
  if (fabs(b) < 0.000001f) return 0.0f; // Avoid divide-by-zero
  return a / b; // Normal division
}

float getMin(const float* arr, int n) {
  float v = arr[0]; // Start with the first value
  for (int i = 1; i < n; i++) if (arr[i] < v) v = arr[i]; // Keep the smallest value
  return v; // Return the minimum
}

float getMax(const float* arr, int n) {
  float v = arr[0]; // Start with the first value
  for (int i = 1; i < n; i++) if (arr[i] > v) v = arr[i]; // Keep the largest value
  return v; // Return the maximum
}

float getMean(const float* arr, int n) {
  float sum = 0.0f; // Running sum
  for (int i = 0; i < n; i++) sum += arr[i]; // Add all values
  return safeDivide(sum, (float)n); // Return average
}

float getStd(const float* arr, int n, float mean) {
  float sum = 0.0f; // Sum of squared differences
  for (int i = 0; i < n; i++) {
    float d = arr[i] - mean; // Difference from the mean
    sum += d * d; // Square and add
  }
  return sqrtf(safeDivide(sum, (float)n)); // Return standard deviation
}

float getAbsChangeSum(const float* arr, int n) {
  float sum = 0.0f; // Running total of absolute changes
  for (int i = 1; i < n; i++) sum += fabs(arr[i] - arr[i - 1]); // Add change from point to point
  return sum; // Return total change
}

float lerp(float a, float b, float t) {
  return a + (b - a) * t; // Linear interpolation between two values
}

float sampleAtFraction(const float* arr, int n, float fraction) {
  if (n <= 1) return arr[0]; // If only one sample exists, return it
  float pos = fraction * (float)(n - 1); // Convert fraction to array position
  int left = (int)floor(pos); // Left index
  int right = (int)ceil(pos); // Right index
  if (left == right) return arr[left]; // If exact point, return it
  float t = pos - (float)left; // Fraction between left and right
  return lerp(arr[left], arr[right], t); // Interpolate the value
}

// ---------------- Feature Writer ----------------
// This writes the 15 stats used for one sensor channel
void writeChannelFeatures(
  float* feat, // Output feature vector
  int startIdx, // Where this channel starts in the vector
  const float* arr, // Input data for one channel
  int n, // Number of frames
  float duration_s // Sequence duration
) {
  float first = arr[0]; // First sample
  float last = arr[n - 1]; // Last sample
  float mean = getMean(arr, n); // Mean value
  float minv = getMin(arr, n); // Minimum value
  float maxv = getMax(arr, n); // Maximum value
  float stdv = getStd(arr, n, mean); // Standard deviation
  float range = maxv - minv; // Range
  float delta = last - first; // Change from first to last
  float absChangeSum = getAbsChangeSum(arr, n); // Total motion/change
  float slope = safeDivide(delta, duration_s); // Average slope across time
  float p0 = sampleAtFraction(arr, n, 0.00f); // Sample at 0%
  float p25 = sampleAtFraction(arr, n, 0.25f); // Sample at 25%
  float p50 = sampleAtFraction(arr, n, 0.50f); // Sample at 50%
  float p75 = sampleAtFraction(arr, n, 0.75f); // Sample at 75%
  float p100 = sampleAtFraction(arr, n, 1.00f); // Sample at 100%

  feat[startIdx + 0] = first; // Store first
  feat[startIdx + 1] = last; // Store last
  feat[startIdx + 2] = mean; // Store mean
  feat[startIdx + 3] = minv; // Store min
  feat[startIdx + 4] = maxv; // Store max
  feat[startIdx + 5] = stdv; // Store std
  feat[startIdx + 6] = range; // Store range
  feat[startIdx + 7] = delta; // Store delta
  feat[startIdx + 8] = absChangeSum; // Store abs change sum
  feat[startIdx + 9] = slope; // Store slope
  feat[startIdx + 10] = p0; // Store p0
  feat[startIdx + 11] = p25; // Store p25
  feat[startIdx + 12] = p50; // Store p50
  feat[startIdx + 13] = p75; // Store p75
  feat[startIdx + 14] = p100; // Store p100
}

// ---------------- Sensor Read Wrapper ----------------
// This reads one raw frame from the ESP32 pins and sensors
RawFrame readRawFrame() {
  RawFrame f; // Create a new frame
  f.timestamp_ms = millis(); // Save current time

  f.hall_thumb  = (float)ads1.readADC_SingleEnded(0); // Read thumb hall sensor
  f.hall_index  = (float)ads2.readADC_SingleEnded(0); // Read index hall sensor
  f.hall_middle = (float)ads2.readADC_SingleEnded(1); // Read middle hall sensor
  f.hall_ring   = (float)ads2.readADC_SingleEnded(2); // Read ring hall sensor
  f.hall_pinky  = (float)ads2.readADC_SingleEnded(3); // Read pinky hall sensor

  imu::Vector<3> accel = bno->getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER); // Read accel
  imu::Vector<3> gyro  = bno->getVector(Adafruit_BNO055::VECTOR_GYROSCOPE); // Read gyro
  imu::Vector<3> mag   = bno->getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER); // Read magnetometer

  f.imu_ax = accel.x(); // Save accel X
  f.imu_ay = accel.y(); // Save accel Y
  f.imu_az = accel.z(); // Save accel Z

  f.imu_gx = gyro.x(); // Save gyro X
  f.imu_gy = gyro.y(); // Save gyro Y
  f.imu_gz = gyro.z(); // Save gyro Z

  f.imu_mx = mag.x(); // Save mag X
  f.imu_my = mag.y(); // Save mag Y
  f.imu_mz = mag.z(); // Save mag Z

  f.contact_p  = (float)digitalRead(PALM); // Read palm contact
  f.contact_i  = (float)digitalRead(INDEX); // Read index contact
  f.contact_m  = (float)digitalRead(MIDDLE); // Read middle contact
  f.contact_um = (float)digitalRead(UNDER_MIDDLE); // Read under-middle contact

  return f; // Return the full frame
}

// ---------------- Sequence To Feature Vector ----------------
// This converts the raw sequence buffer into the exact 284-feature model input
bool buildFeatureVectorFromFrames(const RawFrame* frames, int count, float* feat) {
  if (count < 2) return false; // Need at least 2 frames to compute time-based features

  memset(feat, 0, sizeof(float) * RFModel::NUM_FEATURES); // Clear the output vector

  float duration_s = (float)(frames[count - 1].timestamp_ms - frames[0].timestamp_ms) / 1000.0f; // Sequence duration in seconds
  if (duration_s <= 0.0f) duration_s = 0.000001f; // Prevent divide-by-zero

  feat[IDX_duration_s] = duration_s; // Store duration feature
  feat[IDX_num_frames] = (float)count; // Store number of frames feature

  float hall_thumb[MAX_FRAMES]; // Buffer for thumb channel
  float hall_index[MAX_FRAMES]; // Buffer for index channel
  float hall_middle[MAX_FRAMES]; // Buffer for middle channel
  float hall_ring[MAX_FRAMES]; // Buffer for ring channel
  float hall_pinky[MAX_FRAMES]; // Buffer for pinky channel

  float imu_ax[MAX_FRAMES]; // Buffer for accel X
  float imu_ay[MAX_FRAMES]; // Buffer for accel Y
  float imu_az[MAX_FRAMES]; // Buffer for accel Z

  float imu_gx[MAX_FRAMES]; // Buffer for gyro X
  float imu_gy[MAX_FRAMES]; // Buffer for gyro Y
  float imu_gz[MAX_FRAMES]; // Buffer for gyro Z

  float imu_mx[MAX_FRAMES]; // Buffer for mag X
  float imu_my[MAX_FRAMES]; // Buffer for mag Y
  float imu_mz[MAX_FRAMES]; // Buffer for mag Z

  float contact_p[MAX_FRAMES]; // Buffer for palm contact
  float contact_i[MAX_FRAMES]; // Buffer for index contact
  float contact_m[MAX_FRAMES]; // Buffer for middle contact
  float contact_um[MAX_FRAMES]; // Buffer for under-middle contact

  float accel_mag[MAX_FRAMES]; // Buffer for accel magnitude
  float gyro_mag[MAX_FRAMES]; // Buffer for gyro magnitude
  float mag_mag[MAX_FRAMES]; // Buffer for mag magnitude

  for (int i = 0; i < count; i++) {
    hall_thumb[i] = frames[i].hall_thumb; // Copy thumb data
    hall_index[i] = frames[i].hall_index; // Copy index data
    hall_middle[i] = frames[i].hall_middle; // Copy middle data
    hall_ring[i] = frames[i].hall_ring; // Copy ring data
    hall_pinky[i] = frames[i].hall_pinky; // Copy pinky data

    imu_ax[i] = frames[i].imu_ax; // Copy accel X
    imu_ay[i] = frames[i].imu_ay; // Copy accel Y
    imu_az[i] = frames[i].imu_az; // Copy accel Z

    imu_gx[i] = frames[i].imu_gx; // Copy gyro X
    imu_gy[i] = frames[i].imu_gy; // Copy gyro Y
    imu_gz[i] = frames[i].imu_gz; // Copy gyro Z

    imu_mx[i] = frames[i].imu_mx; // Copy mag X
    imu_my[i] = frames[i].imu_my; // Copy mag Y
    imu_mz[i] = frames[i].imu_mz; // Copy mag Z

    contact_p[i] = frames[i].contact_p; // Copy palm contact
    contact_i[i] = frames[i].contact_i; // Copy index contact
    contact_m[i] = frames[i].contact_m; // Copy middle contact
    contact_um[i] = frames[i].contact_um; // Copy under-middle contact

    accel_mag[i] = sqrtf(
      frames[i].imu_ax * frames[i].imu_ax +
      frames[i].imu_ay * frames[i].imu_ay +
      frames[i].imu_az * frames[i].imu_az
    ); // Compute accel magnitude for this frame

    gyro_mag[i] = sqrtf(
      frames[i].imu_gx * frames[i].imu_gx +
      frames[i].imu_gy * frames[i].imu_gy +
      frames[i].imu_gz * frames[i].imu_gz
    ); // Compute gyro magnitude for this frame

    mag_mag[i] = sqrtf(
      frames[i].imu_mx * frames[i].imu_mx +
      frames[i].imu_my * frames[i].imu_my +
      frames[i].imu_mz * frames[i].imu_mz
    ); // Compute magnetometer magnitude for this frame
  }

  writeChannelFeatures(feat, IDX_hall_thumb_first, hall_thumb, count, duration_s); // Write thumb features
  writeChannelFeatures(feat, IDX_hall_index_first, hall_index, count, duration_s); // Write index features
  writeChannelFeatures(feat, IDX_hall_middle_first, hall_middle, count, duration_s); // Write middle features
  writeChannelFeatures(feat, IDX_hall_ring_first, hall_ring, count, duration_s); // Write ring features
  writeChannelFeatures(feat, IDX_hall_pinky_first, hall_pinky, count, duration_s); // Write pinky features

  writeChannelFeatures(feat, IDX_imu_ax_first, imu_ax, count, duration_s); // Write accel X features
  writeChannelFeatures(feat, IDX_imu_ay_first, imu_ay, count, duration_s); // Write accel Y features
  writeChannelFeatures(feat, IDX_imu_az_first, imu_az, count, duration_s); // Write accel Z features

  writeChannelFeatures(feat, IDX_imu_gx_first, imu_gx, count, duration_s); // Write gyro X features
  writeChannelFeatures(feat, IDX_imu_gy_first, imu_gy, count, duration_s); // Write gyro Y features
  writeChannelFeatures(feat, IDX_imu_gz_first, imu_gz, count, duration_s); // Write gyro Z features

  writeChannelFeatures(feat, IDX_imu_mx_first, imu_mx, count, duration_s); // Write mag X features
  writeChannelFeatures(feat, IDX_imu_my_first, imu_my, count, duration_s); // Write mag Y features
  writeChannelFeatures(feat, IDX_imu_mz_first, imu_mz, count, duration_s); // Write mag Z features

  writeChannelFeatures(feat, IDX_contact_p_first, contact_p, count, duration_s); // Write palm contact features
  writeChannelFeatures(feat, IDX_contact_i_first, contact_i, count, duration_s); // Write index contact features
  writeChannelFeatures(feat, IDX_contact_m_first, contact_m, count, duration_s); // Write middle contact features
  writeChannelFeatures(feat, IDX_contact_um_first, contact_um, count, duration_s); // Write under-middle contact features

  feat[IDX_accel_mag_mean] = getMean(accel_mag, count); // Accel magnitude mean
  feat[IDX_accel_mag_max] = getMax(accel_mag, count); // Accel magnitude max
  feat[IDX_accel_mag_std] = getStd(accel_mag, count, feat[IDX_accel_mag_mean]); // Accel magnitude std
  feat[IDX_accel_mag_abs_change_sum] = getAbsChangeSum(accel_mag, count); // Accel magnitude motion sum

  feat[IDX_gyro_mag_mean] = getMean(gyro_mag, count); // Gyro magnitude mean
  feat[IDX_gyro_mag_max] = getMax(gyro_mag, count); // Gyro magnitude max
  feat[IDX_gyro_mag_std] = getStd(gyro_mag, count, feat[IDX_gyro_mag_mean]); // Gyro magnitude std
  feat[IDX_gyro_mag_abs_change_sum] = getAbsChangeSum(gyro_mag, count); // Gyro magnitude motion sum

  feat[IDX_mag_mag_mean] = getMean(mag_mag, count); // Magnetometer magnitude mean
  feat[IDX_mag_mag_max] = getMax(mag_mag, count); // Magnetometer magnitude max
  feat[IDX_mag_mag_std] = getStd(mag_mag, count, feat[IDX_mag_mag_mean]); // Magnetometer magnitude std
  feat[IDX_mag_mag_abs_change_sum] = getAbsChangeSum(mag_mag, count); // Magnetometer magnitude motion sum

  return true; // Feature vector built successfully
}

// ---------------- Model Call Function ----------------
// This is the main function that calls the random forest
int runGestureModel(const float* featureVector) {
  return RFModel::predict(featureVector); // Return predicted class index
}

// ---------------- Friendly Prediction Wrapper ----------------
// This returns the class label string instead of just the index
const char* predictGestureLabel(const float* featureVector) {
  int classIdx = runGestureModel(featureVector); // Get predicted class number
  return RFModel::class_name(classIdx); // Convert class number to label text
}

// ---------------- End-To-End Wrapper ----------------
// This shows how the ESP32 collects a window, extracts features, and calls the model
bool collectAndPredictGesture() {
  g_frameCount = 0; // Start a fresh sequence buffer

  for (int i = 0; i < MAX_FRAMES; i++) {
    g_frames[g_frameCount] = readRawFrame(); // Read one raw sensor frame
    g_frameCount++; // Count that frame
    delay(SAMPLE_DELAY_MS); // Wait before next sample
  }

  float featureVector[RFModel::NUM_FEATURES]; // Create the 284-feature input buffer

  if (!buildFeatureVectorFromFrames(g_frames, g_frameCount, featureVector)) { // Try to build features
    Serial.println("Feature build failed"); // Print error if it fails
    return false; // Stop here
  }

  int classIdx = runGestureModel(featureVector); // Run the random forest
  const char* label = RFModel::class_name(classIdx); // Convert result to label text

  Serial.print("Predicted class index: "); // Print label for debugging
  Serial.println(classIdx); // Print class index

  Serial.print("Predicted class name: "); // Print label heading
  Serial.println(label); // Print class label

  return true; // Prediction succeeded
}

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200); // Start serial output
  delay(2000); // Give serial monitor time to open

  pinMode(PALM, INPUT_PULLDOWN); // Palm contact input
  pinMode(INDEX, INPUT_PULLDOWN); // Index contact input
  pinMode(MIDDLE, INPUT_PULLDOWN); // Middle contact input
  pinMode(UNDER_MIDDLE, INPUT_PULLDOWN); // Under-middle contact input

  Wire.begin(); // Start I2C
  Wire.setClock(400000); // Use fast I2C

  if (!ads1.begin(0x48, &Wire)) fatalError("ERROR: ADS1115 at 0x48 not found"); // Start first ADC
  if (!ads2.begin(0x49, &Wire)) fatalError("ERROR: ADS1115 at 0x49 not found"); // Start second ADC

  ads1.setGain(GAIN_ONE); // Set ADC gain for ads1
  ads2.setGain(GAIN_ONE); // Set ADC gain for ads2

  if (bno28.begin()) {
    bno = &bno28; // Use IMU at 0x28 if found
  } else if (bno29.begin()) {
    bno = &bno29; // Otherwise use IMU at 0x29 if found
  } else {
    fatalError("ERROR: BNO055 not found at 0x28 or 0x29"); // Stop if no IMU is found
  }

  bno->setExtCrystalUse(true); // Use external crystal for better stability
  delay(1000); // Give the IMU time to settle

  Serial.println("System ready"); // Print ready message
}

// ---------------- Main Loop ----------------
void loop() {
  collectAndPredictGesture(); // Read one gesture window and classify it
  delay(500); // Small pause before the next full gesture read
}