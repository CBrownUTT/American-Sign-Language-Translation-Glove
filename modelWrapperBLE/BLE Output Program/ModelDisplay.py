import asyncio
import threading
import os
from queue import Queue, Empty

from bleak import BleakScanner, BleakClient
from tkinter import Tk, Label, StringVar, Frame
from PIL import Image, ImageTk

# ---------------- BLE settings ----------------

DEVICE_NAME = "ASL_Glove"
CHAR_UUID = "12345678-1234-1234-1234-1234567890ac"

# Folder that contains letter images like A.png, B.png, C.png, etc.
IMAGE_FOLDER = "ASL_Signs"

# ---------------- Shared queue ----------------

message_queue = Queue()

# ---------------- BLE parsing ----------------

def parse_ble_message(message: str):
    """
    Expected format:
    label=A,idx=0,conf=0.91
    """
    parts = {}
    for item in message.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()

    label = parts.get("label", "")
    confidence_str = parts.get("conf", "0")

    try:
        confidence = float(confidence_str)
    except ValueError:
        confidence = 0.0

    return label, confidence

# ---------------- BLE callback ----------------

def notification_handler(characteristic, data: bytearray):
    try:
        message = data.decode("utf-8").strip()
        print("Received:", message)

        label, confidence = parse_ble_message(message)
        message_queue.put((label, confidence))

    except Exception as e:
        print("Error parsing BLE data:", e)

# ---------------- BLE task ----------------

async def ble_task():
    print("Scanning for BLE device...")

    devices = await BleakScanner.discover(timeout=10.0)
    target = None

    for d in devices:
        if d.name == DEVICE_NAME:
            target = d
            break

    if target is None:
        raise RuntimeError(f"Could not find BLE device named '{DEVICE_NAME}'")

    print(f"Found device: {target.name}")

    async with BleakClient(target) as client:
        print("Connected to BLE device")
        await client.start_notify(CHAR_UUID, notification_handler)
        print("Listening for notifications...")

        while True:
            await asyncio.sleep(1)

def run_ble_loop():
    asyncio.run(ble_task())

# ---------------- GUI app ----------------

class ASLDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Glove Prediction Display")
        self.root.geometry("500x600")

        self.current_photo = None

        self.main_frame = Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.title_var = StringVar(value="Waiting for prediction...")
        self.title_label = Label(
            self.main_frame,
            textvariable=self.title_var,
            font=("Arial", 20, "bold")
        )
        self.title_label.pack(pady=10)

        self.image_label = Label(self.main_frame, text="No image loaded", font=("Arial", 14))
        self.image_label.pack(pady=20)

        self.confidence_var = StringVar(value="Confidence: --")
        self.confidence_label = Label(
            self.main_frame,
            textvariable=self.confidence_var,
            font=("Arial", 18)
        )
        self.confidence_label.pack(pady=10)

        self.status_var = StringVar(value="Status: waiting for BLE data")
        self.status_label = Label(
            self.main_frame,
            textvariable=self.status_var,
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        self.root.after(100, self.check_queue)

    def update_display(self, label, confidence):
        self.title_var.set(f"Predicted Letter: {label}")
        self.confidence_var.set(f"Confidence: {confidence:.2%}")
        self.status_var.set("Status: prediction received")

        image_path = os.path.join(IMAGE_FOLDER, f"{label}.png")

        if not os.path.exists(image_path):
            self.image_label.config(
                image="",
                text=f"Image not found:\n{image_path}",
                font=("Arial", 14)
            )
            self.current_photo = None
            return

        try:
            image = Image.open(image_path)

            # Resize image to fit window nicely
            max_size = (350, 350)
            image.thumbnail(max_size)

            photo = ImageTk.PhotoImage(image)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.current_photo = photo

        except Exception as e:
            self.image_label.config(
                image="",
                text=f"Error loading image:\n{e}",
                font=("Arial", 14)
            )
            self.current_photo = None

    def check_queue(self):
        try:
            while True:
                label, confidence = message_queue.get_nowait()
                self.update_display(label, confidence)
        except Empty:
            pass

        self.root.after(100, self.check_queue)

# ---------------- Main ----------------

def main():
    ble_thread = threading.Thread(target=run_ble_loop, daemon=True)
    ble_thread.start()

    root = Tk()
    app = ASLDisplayApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()