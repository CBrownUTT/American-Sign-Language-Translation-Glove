import os
import math
import joblib

JOBLIB_PATH = "training_outputs/asl_random_forest.joblib"
OUT_DIR = "cpp_export"
MODEL_BASENAME = "asl_rf_model"

EXPECTED_SENSOR_COLUMNS = [
    "hall_thumb", "hall_index", "hall_middle", "hall_ring", "hall_pinky",
    "imu_ax", "imu_ay", "imu_az",
    "imu_gx", "imu_gy", "imu_gz",
    "imu_ex", "imu_ey", "imu_ez",
    "contact_p", "contact_i", "contact_m", "contact_um"
]

def cpp_escape_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def cpp_float(x) -> str:
    x = float(x)

    if math.isnan(x):
        return "NAN"
    if math.isinf(x):
        return "INFINITY" if x > 0 else "-INFINITY"

    s = f"{x:.9g}"

    # If the formatted value has no decimal point or exponent,
    # add .0 so Arduino treats it as a float literal.
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"

    return s + "f"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_header_guard(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch.upper() if ch.isalnum() else "_")
    return "".join(out) + "_H"

def validate_bundle(bundle):
    required = {"model", "feature_columns", "sensor_columns", "class_labels"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Joblib bundle is missing keys: {sorted(missing)}")

    model = bundle["model"]
    if model.__class__.__name__ != "RandomForestClassifier":
        raise TypeError(f"Expected RandomForestClassifier, got {type(model)}")

    feature_columns = list(bundle["feature_columns"])
    sensor_columns = list(bundle["sensor_columns"])
    class_labels = list(bundle["class_labels"])

    if len(feature_columns) != model.n_features_in_:
        raise ValueError(
            f"Feature column count mismatch: {len(feature_columns)} vs model.n_features_in_={model.n_features_in_}"
        )

    return model, feature_columns, sensor_columns, class_labels

def build_expected_feature_columns(sensor_columns):
    cols = ["duration_s", "num_frames"]
    stats_suffixes = [
        "first", "last", "mean", "min", "max", "std",
        "range", "delta", "abs_change_sum", "slope",
        "p0", "p25", "p50", "p75", "p100"
    ]

    for s in sensor_columns:
        for suffix in stats_suffixes:
            cols.append(f"{s}_{suffix}")

    cols += [
        "accel_mag_mean", "accel_mag_max", "accel_mag_std", "accel_mag_abs_change_sum",
        "gyro_mag_mean", "gyro_mag_max", "gyro_mag_std", "gyro_mag_abs_change_sum",
        "euler_mag_mean", "euler_mag_max", "euler_mag_std", "euler_mag_abs_change_sum",
    ]
    return cols

def write_feature_builder_hpp(path, feature_columns, sensor_columns):
    guard = safe_header_guard("feature_builder")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'''#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stddef.h>

namespace ASLModel {{

static constexpr int SENSOR_COUNT = {len(sensor_columns)};
static constexpr int FEATURE_COUNT = {len(feature_columns)};

struct Frame {{
    float sensor[SENSOR_COUNT];
    uint32_t timestamp_ms;
}};

extern const char* const kSensorNames[SENSOR_COUNT];
extern const char* const kFeatureNames[FEATURE_COUNT];

bool build_feature_vector(const Frame* frames, int frame_count, float* out_features);

}}  // namespace ASLModel

#endif
''')

def write_feature_builder_cpp(path, feature_columns, sensor_columns):
    sensor_name_array = ",\n    ".join(f'"{cpp_escape_string(x)}"' for x in sensor_columns)
    feature_name_array = ",\n    ".join(f'"{cpp_escape_string(x)}"' for x in feature_columns)

    sidx = {name: i for i, name in enumerate(sensor_columns)}
    required_names = [
        "imu_ax", "imu_ay", "imu_az",
        "imu_gx", "imu_gy", "imu_gz",
        "imu_ex", "imu_ey", "imu_ez",
    ]
    for name in required_names:
        if name not in sidx:
            raise ValueError(f"Required sensor '{name}' not found in sensor_columns")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f'''#include "feature_builder.h"
#include <math.h>

namespace ASLModel {{

const char* const kSensorNames[SENSOR_COUNT] = {{
    {sensor_name_array}
}};

const char* const kFeatureNames[FEATURE_COUNT] = {{
    {feature_name_array}
}};

static float interp_position(const float* arr, int n, float pos01) {{
    if (n <= 0) return 0.0f;
    if (n == 1) return arr[0];

    float x = pos01 * (float)(n - 1);
    int i0 = (int)floorf(x);
    int i1 = i0 + 1;

    if (i0 < 0) i0 = 0;
    if (i1 >= n) i1 = n - 1;

    float t = x - (float)i0;
    return arr[i0] * (1.0f - t) + arr[i1] * t;
}}

static void compute_basic_stats(
    const float* arr,
    int n,
    float duration_s,
    float& first,
    float& last,
    float& mean,
    float& min_v,
    float& max_v,
    float& std_v,
    float& range_v,
    float& delta,
    float& abs_change_sum,
    float& slope,
    float& p0,
    float& p25,
    float& p50,
    float& p75,
    float& p100
) {{
    first = arr[0];
    last = arr[n - 1];

    float sum = 0.0f;
    min_v = arr[0];
    max_v = arr[0];
    abs_change_sum = 0.0f;

    for (int i = 0; i < n; ++i) {{
        float v = arr[i];
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (i > 0) abs_change_sum += fabsf(arr[i] - arr[i - 1]);
    }}

    mean = sum / (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; ++i) {{
        float d = arr[i] - mean;
        var += d * d;
    }}
    var /= (float)n;
    std_v = sqrtf(var);

    range_v = max_v - min_v;
    delta = last - first;
    slope = delta / duration_s;

    p0 = interp_position(arr, n, 0.00f);
    p25 = interp_position(arr, n, 0.25f);
    p50 = interp_position(arr, n, 0.50f);
    p75 = interp_position(arr, n, 0.75f);
    p100 = interp_position(arr, n, 1.00f);
}}

bool build_feature_vector(const Frame* frames, int frame_count, float* out_features) {{
    if (frames == nullptr || out_features == nullptr) return false;
    if (frame_count < 2) return false;
    if (frame_count > 128) return false;

    float t0 = (float)frames[0].timestamp_ms / 1000.0f;
    float t1 = (float)frames[frame_count - 1].timestamp_ms / 1000.0f;
    float duration_s = t1 - t0;
    if (duration_s < 1e-6f) duration_s = 1e-6f;

    int out_i = 0;
    out_features[out_i++] = duration_s;
    out_features[out_i++] = (float)frame_count;

    float tmp[128];

    for (int sensor_idx = 0; sensor_idx < SENSOR_COUNT; ++sensor_idx) {{
        for (int i = 0; i < frame_count; ++i) {{
            tmp[i] = frames[i].sensor[sensor_idx];
        }}

        float first, last, mean, min_v, max_v, std_v, range_v, delta, abs_change_sum, slope;
        float p0, p25, p50, p75, p100;

        compute_basic_stats(
            tmp, frame_count, duration_s,
            first, last, mean, min_v, max_v, std_v,
            range_v, delta, abs_change_sum, slope,
            p0, p25, p50, p75, p100
        );

        out_features[out_i++] = first;
        out_features[out_i++] = last;
        out_features[out_i++] = mean;
        out_features[out_i++] = min_v;
        out_features[out_i++] = max_v;
        out_features[out_i++] = std_v;
        out_features[out_i++] = range_v;
        out_features[out_i++] = delta;
        out_features[out_i++] = abs_change_sum;
        out_features[out_i++] = slope;
        out_features[out_i++] = p0;
        out_features[out_i++] = p25;
        out_features[out_i++] = p50;
        out_features[out_i++] = p75;
        out_features[out_i++] = p100;
    }}

    float accel_mag[128];
    float gyro_mag[128];
    float euler_mag[128];

    for (int i = 0; i < frame_count; ++i) {{
        const float ax = frames[i].sensor[{sidx["imu_ax"]}];
        const float ay = frames[i].sensor[{sidx["imu_ay"]}];
        const float az = frames[i].sensor[{sidx["imu_az"]}];

        const float gx = frames[i].sensor[{sidx["imu_gx"]}];
        const float gy = frames[i].sensor[{sidx["imu_gy"]}];
        const float gz = frames[i].sensor[{sidx["imu_gz"]}];

        const float ex = frames[i].sensor[{sidx["imu_ex"]}];
        const float ey = frames[i].sensor[{sidx["imu_ey"]}];
        const float ez = frames[i].sensor[{sidx["imu_ez"]}];

        accel_mag[i] = sqrtf(ax * ax + ay * ay + az * az);
        gyro_mag[i] = sqrtf(gx * gx + gy * gy + gz * gz);
        euler_mag[i] = sqrtf(ex * ex + ey * ey + ez * ez);
    }}

    auto write_mag_stats = [&](const float* arr) {{
        float sum = 0.0f;
        float max_v = arr[0];
        float abs_change_sum = 0.0f;

        for (int i = 0; i < frame_count; ++i) {{
            float v = arr[i];
            sum += v;
            if (v > max_v) max_v = v;
            if (i > 0) abs_change_sum += fabsf(arr[i] - arr[i - 1]);
        }}

        float mean = sum / (float)frame_count;
        float var = 0.0f;
        for (int i = 0; i < frame_count; ++i) {{
            float d = arr[i] - mean;
            var += d * d;
        }}
        var /= (float)frame_count;
        float std_v = sqrtf(var);

        out_features[out_i++] = mean;
        out_features[out_i++] = max_v;
        out_features[out_i++] = std_v;
        out_features[out_i++] = abs_change_sum;
    }};

    write_mag_stats(accel_mag);
    write_mag_stats(gyro_mag);
    write_mag_stats(euler_mag);

    return out_i == FEATURE_COUNT;
}}

}}  // namespace ASLModel
''')

def write_model_hpp(path, model_name, class_labels, feature_count, n_trees):
    guard = safe_header_guard(model_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'''#ifndef {guard}
#define {guard}

namespace ASLModel {{

static constexpr int kFeatureCount = {feature_count};
static constexpr int kClassCount = {len(class_labels)};
static constexpr int kTreeCount = {n_trees};

extern const char* const kClassLabels[kClassCount];

int predict(const float* features);
void predict_proba(const float* features, float* out_proba);

}}  // namespace ASLModel

#endif
''')

def emit_tree_function(tree, tree_idx, class_count):
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    lines = []
    lines.append(f"static void tree_{tree_idx}(const float* f, float* votes) {{")

    def recurse(node, indent):
        sp = " " * indent
        left = children_left[node]
        right = children_right[node]

        if left == -1 and right == -1:
            vals = value[node][0]
            for c in range(class_count):
                v = float(vals[c])
                if v != 0.0:
                    lines.append(f"{sp}votes[{c}] += {cpp_float(v)};")
            return

        feat_idx = int(feature[node])
        thr = cpp_float(threshold[node])

        lines.append(f"{sp}if (f[{feat_idx}] <= {thr}) {{")
        recurse(left, indent + 4)
        lines.append(f"{sp}}} else {{")
        recurse(right, indent + 4)
        lines.append(f"{sp}}}")

    recurse(0, 4)
    lines.append("}")
    return "\n".join(lines)

def write_model_cpp(path, model, class_labels):
    with open(path, "w", encoding="utf-8") as f:
        f.write('#include "asl_rf_model.h"\n\n')
        f.write("namespace ASLModel {\n\n")
        labels = ",\n    ".join(f'"{cpp_escape_string(x)}"' for x in class_labels)
        f.write(f"const char* const kClassLabels[kClassCount] = {{\n    {labels}\n}};\n\n")

        for i, est in enumerate(model.estimators_):
            f.write(emit_tree_function(est.tree_, i, len(class_labels)))
            f.write("\n\n")

        f.write('''void predict_proba(const float* features, float* out_proba) {
    float votes[kClassCount];
    for (int i = 0; i < kClassCount; ++i) votes[i] = 0.0f;
''')
        for i in range(len(model.estimators_)):
            f.write(f"    tree_{i}(features, votes);\n")

        f.write('''
    float total = 0.0f;
    for (int i = 0; i < kClassCount; ++i) total += votes[i];
    if (total <= 0.0f) total = 1.0f;

    for (int i = 0; i < kClassCount; ++i) {
        out_proba[i] = votes[i] / total;
    }
}

int predict(const float* features) {
    float proba[kClassCount];
    predict_proba(features, proba);

    int best_idx = 0;
    float best_val = proba[0];
    for (int i = 1; i < kClassCount; ++i) {
        if (proba[i] > best_val) {
            best_val = proba[i];
            best_idx = i;
        }
    }
    return best_idx;
}

}  // namespace ASLModel
''')

def write_wrapper(path):
    wrapper_code = r'''#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

#include "feature_builder.h"
#include "asl_rf_model.h"

using namespace ASLModel;

#define SDA_PIN A4
#define SCL_PIN A5

#define PIN_CONTACT_P   A0
#define PIN_CONTACT_I   A1
#define PIN_CONTACT_M   A2
#define PIN_CONTACT_UM  A3

Adafruit_ADS1115 ads1;
Adafruit_ADS1115 ads2;
Adafruit_BNO055 bno28(55, 0x28);
Adafruit_BNO055 bno29(56, 0x29);
Adafruit_BNO055* bno = nullptr;

static constexpr int FRAME_COUNT = 15;
static constexpr int SAMPLE_DELAY_MS = 20;

Frame g_frames[FRAME_COUNT];
float g_features[FEATURE_COUNT];
float g_proba[kClassCount];

float g_ax = 0.0f, g_ay = 0.0f, g_az = 0.0f;
float g_gx = 0.0f, g_gy = 0.0f, g_gz = 0.0f;
float g_ex = 0.0f, g_ey = 0.0f, g_ez = 0.0f;

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

void setup() {
  Serial.begin(115200);
  delay(2000);

  if (!initHardware()) {
    fatalError("Hardware init failed");
  }

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

  printTop3();
  delay(1000);
}
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(wrapper_code)

def write_metadata_txt(path, model, feature_columns, sensor_columns, class_labels):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Export summary\n")
        f.write("==============\n")
        f.write(f"Estimator type: {type(model).__name__}\n")
        f.write(f"Number of trees: {len(model.estimators_)}\n")
        f.write(f"Number of features: {len(feature_columns)}\n")
        f.write(f"Number of raw sensors: {len(sensor_columns)}\n")
        f.write(f"Number of classes: {len(class_labels)}\n\n")

        f.write("Sensor column order used for feature building:\n")
        for i, s in enumerate(sensor_columns):
            f.write(f"{i:2d}: {s}\n")

        f.write("\nClass labels:\n")
        for i, c in enumerate(class_labels):
            f.write(f"{i:2d}: {c}\n")

        f.write("\nFeature columns:\n")
        for i, c in enumerate(feature_columns):
            f.write(f"{i:3d}: {c}\n")

def main():
    ensure_dir(OUT_DIR)

    bundle = joblib.load(JOBLIB_PATH)
    model, feature_columns, sensor_columns, class_labels = validate_bundle(bundle)

    if sensor_columns != EXPECTED_SENSOR_COLUMNS:
        raise ValueError(
            "Saved sensor_columns do not match the expected Euler-based layout.\n"
            f"Saved: {sensor_columns}\n"
            f"Expected: {EXPECTED_SENSOR_COLUMNS}"
        )

    expected = build_expected_feature_columns(sensor_columns)
    if feature_columns != expected:
        raise ValueError(
            "Saved feature_columns do not match the expected feature builder layout.\n"
            "This exporter stops here to avoid generating wrong C++."
        )

    write_feature_builder_hpp(os.path.join(OUT_DIR, "feature_builder.h"), feature_columns, sensor_columns)
    write_feature_builder_cpp(os.path.join(OUT_DIR, "feature_builder.cpp"), feature_columns, sensor_columns)
    write_model_hpp(os.path.join(OUT_DIR, f"{MODEL_BASENAME}.h"), MODEL_BASENAME, class_labels, len(feature_columns), len(model.estimators_))
    write_model_cpp(os.path.join(OUT_DIR, f"{MODEL_BASENAME}.cpp"), model, class_labels)
    write_wrapper(os.path.join(OUT_DIR, "asl_rf_wrapper_euler.ino"))
    write_metadata_txt(os.path.join(OUT_DIR, "export_metadata.txt"), model, feature_columns, sensor_columns, class_labels)

    print("Export complete.")
    print(f"Files written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
