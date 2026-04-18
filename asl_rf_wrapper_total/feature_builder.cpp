#include "feature_builder.h"
#include <math.h>

namespace ASLModel {

const char* const kSensorNames[SENSOR_COUNT] = {
    "hall_thumb",
    "hall_index",
    "hall_middle",
    "hall_ring",
    "hall_pinky",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "imu_ex",
    "imu_ey",
    "imu_ez",
    "contact_p",
    "contact_i",
    "contact_m",
    "contact_um"
};

const char* const kFeatureNames[FEATURE_COUNT] = {
    "duration_s",
    "num_frames",
    "hall_thumb_first",
    "hall_thumb_last",
    "hall_thumb_mean",
    "hall_thumb_min",
    "hall_thumb_max",
    "hall_thumb_std",
    "hall_thumb_range",
    "hall_thumb_delta",
    "hall_thumb_abs_change_sum",
    "hall_thumb_slope",
    "hall_thumb_p0",
    "hall_thumb_p25",
    "hall_thumb_p50",
    "hall_thumb_p75",
    "hall_thumb_p100",
    "hall_index_first",
    "hall_index_last",
    "hall_index_mean",
    "hall_index_min",
    "hall_index_max",
    "hall_index_std",
    "hall_index_range",
    "hall_index_delta",
    "hall_index_abs_change_sum",
    "hall_index_slope",
    "hall_index_p0",
    "hall_index_p25",
    "hall_index_p50",
    "hall_index_p75",
    "hall_index_p100",
    "hall_middle_first",
    "hall_middle_last",
    "hall_middle_mean",
    "hall_middle_min",
    "hall_middle_max",
    "hall_middle_std",
    "hall_middle_range",
    "hall_middle_delta",
    "hall_middle_abs_change_sum",
    "hall_middle_slope",
    "hall_middle_p0",
    "hall_middle_p25",
    "hall_middle_p50",
    "hall_middle_p75",
    "hall_middle_p100",
    "hall_ring_first",
    "hall_ring_last",
    "hall_ring_mean",
    "hall_ring_min",
    "hall_ring_max",
    "hall_ring_std",
    "hall_ring_range",
    "hall_ring_delta",
    "hall_ring_abs_change_sum",
    "hall_ring_slope",
    "hall_ring_p0",
    "hall_ring_p25",
    "hall_ring_p50",
    "hall_ring_p75",
    "hall_ring_p100",
    "hall_pinky_first",
    "hall_pinky_last",
    "hall_pinky_mean",
    "hall_pinky_min",
    "hall_pinky_max",
    "hall_pinky_std",
    "hall_pinky_range",
    "hall_pinky_delta",
    "hall_pinky_abs_change_sum",
    "hall_pinky_slope",
    "hall_pinky_p0",
    "hall_pinky_p25",
    "hall_pinky_p50",
    "hall_pinky_p75",
    "hall_pinky_p100",
    "imu_ax_first",
    "imu_ax_last",
    "imu_ax_mean",
    "imu_ax_min",
    "imu_ax_max",
    "imu_ax_std",
    "imu_ax_range",
    "imu_ax_delta",
    "imu_ax_abs_change_sum",
    "imu_ax_slope",
    "imu_ax_p0",
    "imu_ax_p25",
    "imu_ax_p50",
    "imu_ax_p75",
    "imu_ax_p100",
    "imu_ay_first",
    "imu_ay_last",
    "imu_ay_mean",
    "imu_ay_min",
    "imu_ay_max",
    "imu_ay_std",
    "imu_ay_range",
    "imu_ay_delta",
    "imu_ay_abs_change_sum",
    "imu_ay_slope",
    "imu_ay_p0",
    "imu_ay_p25",
    "imu_ay_p50",
    "imu_ay_p75",
    "imu_ay_p100",
    "imu_az_first",
    "imu_az_last",
    "imu_az_mean",
    "imu_az_min",
    "imu_az_max",
    "imu_az_std",
    "imu_az_range",
    "imu_az_delta",
    "imu_az_abs_change_sum",
    "imu_az_slope",
    "imu_az_p0",
    "imu_az_p25",
    "imu_az_p50",
    "imu_az_p75",
    "imu_az_p100",
    "imu_gx_first",
    "imu_gx_last",
    "imu_gx_mean",
    "imu_gx_min",
    "imu_gx_max",
    "imu_gx_std",
    "imu_gx_range",
    "imu_gx_delta",
    "imu_gx_abs_change_sum",
    "imu_gx_slope",
    "imu_gx_p0",
    "imu_gx_p25",
    "imu_gx_p50",
    "imu_gx_p75",
    "imu_gx_p100",
    "imu_gy_first",
    "imu_gy_last",
    "imu_gy_mean",
    "imu_gy_min",
    "imu_gy_max",
    "imu_gy_std",
    "imu_gy_range",
    "imu_gy_delta",
    "imu_gy_abs_change_sum",
    "imu_gy_slope",
    "imu_gy_p0",
    "imu_gy_p25",
    "imu_gy_p50",
    "imu_gy_p75",
    "imu_gy_p100",
    "imu_gz_first",
    "imu_gz_last",
    "imu_gz_mean",
    "imu_gz_min",
    "imu_gz_max",
    "imu_gz_std",
    "imu_gz_range",
    "imu_gz_delta",
    "imu_gz_abs_change_sum",
    "imu_gz_slope",
    "imu_gz_p0",
    "imu_gz_p25",
    "imu_gz_p50",
    "imu_gz_p75",
    "imu_gz_p100",
    "imu_ex_first",
    "imu_ex_last",
    "imu_ex_mean",
    "imu_ex_min",
    "imu_ex_max",
    "imu_ex_std",
    "imu_ex_range",
    "imu_ex_delta",
    "imu_ex_abs_change_sum",
    "imu_ex_slope",
    "imu_ex_p0",
    "imu_ex_p25",
    "imu_ex_p50",
    "imu_ex_p75",
    "imu_ex_p100",
    "imu_ey_first",
    "imu_ey_last",
    "imu_ey_mean",
    "imu_ey_min",
    "imu_ey_max",
    "imu_ey_std",
    "imu_ey_range",
    "imu_ey_delta",
    "imu_ey_abs_change_sum",
    "imu_ey_slope",
    "imu_ey_p0",
    "imu_ey_p25",
    "imu_ey_p50",
    "imu_ey_p75",
    "imu_ey_p100",
    "imu_ez_first",
    "imu_ez_last",
    "imu_ez_mean",
    "imu_ez_min",
    "imu_ez_max",
    "imu_ez_std",
    "imu_ez_range",
    "imu_ez_delta",
    "imu_ez_abs_change_sum",
    "imu_ez_slope",
    "imu_ez_p0",
    "imu_ez_p25",
    "imu_ez_p50",
    "imu_ez_p75",
    "imu_ez_p100",
    "contact_p_first",
    "contact_p_last",
    "contact_p_mean",
    "contact_p_min",
    "contact_p_max",
    "contact_p_std",
    "contact_p_range",
    "contact_p_delta",
    "contact_p_abs_change_sum",
    "contact_p_slope",
    "contact_p_p0",
    "contact_p_p25",
    "contact_p_p50",
    "contact_p_p75",
    "contact_p_p100",
    "contact_i_first",
    "contact_i_last",
    "contact_i_mean",
    "contact_i_min",
    "contact_i_max",
    "contact_i_std",
    "contact_i_range",
    "contact_i_delta",
    "contact_i_abs_change_sum",
    "contact_i_slope",
    "contact_i_p0",
    "contact_i_p25",
    "contact_i_p50",
    "contact_i_p75",
    "contact_i_p100",
    "contact_m_first",
    "contact_m_last",
    "contact_m_mean",
    "contact_m_min",
    "contact_m_max",
    "contact_m_std",
    "contact_m_range",
    "contact_m_delta",
    "contact_m_abs_change_sum",
    "contact_m_slope",
    "contact_m_p0",
    "contact_m_p25",
    "contact_m_p50",
    "contact_m_p75",
    "contact_m_p100",
    "contact_um_first",
    "contact_um_last",
    "contact_um_mean",
    "contact_um_min",
    "contact_um_max",
    "contact_um_std",
    "contact_um_range",
    "contact_um_delta",
    "contact_um_abs_change_sum",
    "contact_um_slope",
    "contact_um_p0",
    "contact_um_p25",
    "contact_um_p50",
    "contact_um_p75",
    "contact_um_p100",
    "accel_mag_mean",
    "accel_mag_max",
    "accel_mag_std",
    "accel_mag_abs_change_sum",
    "gyro_mag_mean",
    "gyro_mag_max",
    "gyro_mag_std",
    "gyro_mag_abs_change_sum",
    "euler_mag_mean",
    "euler_mag_max",
    "euler_mag_std",
    "euler_mag_abs_change_sum"
};

static float interp_position(const float* arr, int n, float pos01) {
    if (n <= 0) return 0.0f;
    if (n == 1) return arr[0];

    float x = pos01 * (float)(n - 1);
    int i0 = (int)floorf(x);
    int i1 = i0 + 1;

    if (i0 < 0) i0 = 0;
    if (i1 >= n) i1 = n - 1;

    float t = x - (float)i0;
    return arr[i0] * (1.0f - t) + arr[i1] * t;
}

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
) {
    first = arr[0];
    last = arr[n - 1];

    float sum = 0.0f;
    min_v = arr[0];
    max_v = arr[0];
    abs_change_sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float v = arr[i];
        sum += v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (i > 0) abs_change_sum += fabsf(arr[i] - arr[i - 1]);
    }

    mean = sum / (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = arr[i] - mean;
        var += d * d;
    }
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
}

bool build_feature_vector(const Frame* frames, int frame_count, float* out_features) {
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

    for (int sensor_idx = 0; sensor_idx < SENSOR_COUNT; ++sensor_idx) {
        for (int i = 0; i < frame_count; ++i) {
            tmp[i] = frames[i].sensor[sensor_idx];
        }

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
    }

    float accel_mag[128];
    float gyro_mag[128];
    float euler_mag[128];

    for (int i = 0; i < frame_count; ++i) {
        const float ax = frames[i].sensor[5];
        const float ay = frames[i].sensor[6];
        const float az = frames[i].sensor[7];

        const float gx = frames[i].sensor[8];
        const float gy = frames[i].sensor[9];
        const float gz = frames[i].sensor[10];

        const float ex = frames[i].sensor[11];
        const float ey = frames[i].sensor[12];
        const float ez = frames[i].sensor[13];

        accel_mag[i] = sqrtf(ax * ax + ay * ay + az * az);
        gyro_mag[i] = sqrtf(gx * gx + gy * gy + gz * gz);
        euler_mag[i] = sqrtf(ex * ex + ey * ey + ez * ez);
    }

    auto write_mag_stats = [&](const float* arr) {
        float sum = 0.0f;
        float max_v = arr[0];
        float abs_change_sum = 0.0f;

        for (int i = 0; i < frame_count; ++i) {
            float v = arr[i];
            sum += v;
            if (v > max_v) max_v = v;
            if (i > 0) abs_change_sum += fabsf(arr[i] - arr[i - 1]);
        }

        float mean = sum / (float)frame_count;
        float var = 0.0f;
        for (int i = 0; i < frame_count; ++i) {
            float d = arr[i] - mean;
            var += d * d;
        }
        var /= (float)frame_count;
        float std_v = sqrtf(var);

        out_features[out_i++] = mean;
        out_features[out_i++] = max_v;
        out_features[out_i++] = std_v;
        out_features[out_i++] = abs_change_sum;
    };

    write_mag_stats(accel_mag);
    write_mag_stats(gyro_mag);
    write_mag_stats(euler_mag);

    return out_i == FEATURE_COUNT;
}

}  // namespace ASLModel
