import cv2
import numpy as np
import pyfirmata
import time

port = "/dev/ttyUSB0"
board = pyfirmata.Arduino(port)

LED_pins = {
    1: board.digital[7],
    2: board.digital[8],
    3: board.digital[9],
    4: board.digital[10]
}

led_status = {1: False, 2: False, 3: False, 4: False}
led_timers = {1: 0, 2: 0, 3: 0, 4: 0}

TARGET_SIZE = (640, 360)
cap = cv2.VideoCapture("/dev/video2")

def select_pin(i):
    return i + 1

def classify_object(r, g, b):
    ml_g, mu_g = [5, 60, 40], [60, 120, 100]
    ml_b, mu_b = [130, 10, 10], [190, 60, 60]
    
    if ml_g[0] < b < mu_g[0] and ml_g[1] < g < mu_g[1] and ml_g[2] < r < mu_g[2]:
        return "Green Object"
    elif ml_b[0] < b < mu_b[0] and ml_b[1] < g < mu_b[1] and ml_b[2] < r < mu_b[2]:
        return "Blue Object"
    return "Unknown"

def control_led(cls, i):
    global led_timers, led_status
    
    current_time = time.time()

    if cls in ["Green Object", "Blue Object"]:
        if not led_status[4]:
            LED_pins[4].write(1)
            led_status[4] = True
            led_timers[4] = current_time + 1.5

        if cls == "Green Object":
            if not led_status[i]:
                LED_pins[i].write(1)
                led_status[i] = True
                led_timers[i] = current_time + 1.5

while True:
    ret, im = cap.read()
    im = cv2.flip(im, 1)
    
    if not ret:
        continue

    im_resized = cv2.resize(im, TARGET_SIZE)
    im_flipped = cv2.flip(im_resized, 1)

    h, w = im_flipped.shape[:2]
    positions = [(int(w/4), int(h/2)), (int(w/2), int(h/2)), (int(w*3/4), int(h/2))]

    for i, (px, py) in enumerate(positions):
        cropped = im_flipped[py-8:py+9, px-8:px+9, :]
        b, g, r = int(np.mean(cropped[:, :, 0])), int(np.mean(cropped[:, :, 1])), int(np.mean(cropped[:, :, 2]))

        pin = select_pin(i)
        classification = classify_object(r, g, b)
        
        control_led(classification, pin)

        cv2.rectangle(im_flipped, (px-8, py-8), (px+8, py+8), (255, 255, 255), 2)
        cv2.putText(im_flipped, classification, (px - 40, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    current_time = time.time()
    for pin in led_timers:
        if led_status[pin] and current_time > led_timers[pin]:
            LED_pins[pin].write(0)
            led_status[pin] = False
            led_timers[pin] = 0

    cv2.imshow('camera', im_flipped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()