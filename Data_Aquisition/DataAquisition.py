import csv
import time
import uuid
import os
import serial # pip install pyserial
import keyboard  # pip install keyboard                        

SERIAL_PORT = "COM6"   # Change this to your real COM port
BAUD_RATE = 115200
OUTPUT_CSV = "glove_sequence_data_total.csv"

# Must match the Arduino .ino header exactly
FEATURE_COLUMNS = [
    "hall_thumb", "hall_index", "hall_middle", "hall_ring", "hall_pinky",
    "imu_ax", "imu_ay", "imu_az",
    "imu_gx", "imu_gy", "imu_gz",
    "imu_ex", "imu_ey", "imu_ez",
    "contact_p", "contact_i", "contact_m", "contact_um"
]

CSV_COLUMNS = ["sequence_id", "label", "frame_idx", "timestamp_ms"] + FEATURE_COLUMNS


def ensure_csv_exists(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def parse_sensor_line(line):
    parts = line.strip().split(",")

    if not parts:
        return None

    # Skip the Arduino CSV header
    if parts[0] == "hall_thumb":
        return None

    if len(parts) != len(FEATURE_COLUMNS):
        return None

    try:
        return [float(x.strip()) for x in parts]
    except ValueError:
        return None


def wait_for_space():
    print("Press SPACE to start the next capture...")
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN and event.name == "space":
            break


def record_sequence(ser, label, duration_s, capture_num, total_captures):
    sequence_id = str(uuid.uuid4())[:8]
    print(
        f"\nRecording capture {capture_num}/{total_captures} "
        f"for label '{label}' for {duration_s:.2f} seconds"
    )
    print(f"Sequence ID: {sequence_id}")

    start_time = time.time()
    frame_idx = 0
    rows_written = 0

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        while time.time() - start_time < duration_s:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            parsed = parse_sensor_line(raw)
            if parsed is None:
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            writer.writerow([sequence_id, label, frame_idx, timestamp_ms] + parsed)

            frame_idx += 1
            rows_written += 1

    print(f"Saved {rows_written} frames for sequence {sequence_id}")


def record_multiple_sequences(ser, label, duration_s, num_captures):
    for i in range(1, num_captures + 1):
        wait_for_space()
        ser.reset_input_buffer()
        time.sleep(0.1)
        record_sequence(ser, label, duration_s, i, num_captures)

    print(f"\nFinished recording {num_captures} capture(s) for label '{label}'.")


def main():
    ensure_csv_exists(OUTPUT_CSV)

    print(f"Opening serial port {SERIAL_PORT} at {BAUD_RATE}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    print("\nGesture recording")
    print("Suggested durations:")
    print("  Static signs (A, B, C, etc.): 1.0")
    print("  Dynamic signs (J, Z): 2.0")
    print("Type 'quit' as the label to exit.\n")

    try:
        while True:
            label = input("Gesture label: ").strip()
            if label.lower() == "quit":
                break
            if not label:
                print("Label cannot be empty.")
                continue

            try:
                num_captures = int(input("How many captures for this label? ").strip())
                if num_captures <= 0:
                    print("Please enter a number greater than 0.")
                    continue
            except ValueError:
                print("Please enter a valid whole number.")
                continue

            try:
                duration_s = float(input("Recording duration (seconds): ").strip())
                if duration_s <= 0:
                    print("Please enter a duration greater than 0.")
                    continue
            except ValueError:
                print("Please enter a valid number.")
                continue

            print(
                f"\nReady to record label '{label}' "
                f"{num_captures} time(s) for {duration_s:.2f} seconds each."
            )
            print("You will press SPACE before each capture starts.")

            record_multiple_sequences(ser, label, duration_s, num_captures)

    finally:
        ser.close()
        print(f"\nDone. Data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
