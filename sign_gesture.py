import csv
import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from tkinter import *
from PIL import Image, ImageTk

expression = ' '

print('checking network ...')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.4)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    try:

        def cls():
            global expression
            expression = ' '
            e.delete(0, "end")

        def enter_text(x):
            global expression
            e.delete(0, "end")
            expression = expression + str(x)
            e.insert(0, expression)

        args = get_args()
        cap = cv.VideoCapture(args.device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

        # ------------ GUI SETUP -----------------
        root = Tk()
        root.title("Sign.AI")
        root.geometry("1200x600")
        root.config(bg="#1C1C1C")

        photo = PhotoImage(file="images/logo.png")
        root.iconphoto(False, photo)

        Label(root, text="Sign Language Detection",
              font=("times new roman", 30, "bold"),
              bg="#1C1C1C", fg="#78F093").grid(row=0, column=0, columnspan=4)

        # Video Frame
        f1 = LabelFrame(root, bg="#DEDEDE")
        f1.grid(row=1, column=1, columnspan=2)
        l1 = Label(f1, bg="#1C1C1C")
        l1.grid(row=0, column=0)

        # Input box
        e = Entry(root, width=53, fg="#DEDEDE",
                  font=("Courier", 18, "bold"), bg="#1C1C1C")
        e.grid(row=2, column=1, columnspan=2)

        Button(root, text='Clear', height=1, width=20, bd=3,
               command=cls).grid(row=3, column=1, columnspan=2)

        # Right-side ISL chart
        image = Image.open("images/signstpw.png").resize((320, 540)).convert("RGBA")
        photo2 = ImageTk.PhotoImage(image)
        Label(root, image=photo2, bg="#1C1C1C").grid(row=1, column=3, rowspan=3)

        # ------------ MODEL SETUP -----------------
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=args.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

        mode = 0
        times_captured = 0
        prevchar = ' '
        count = 0
        spcount = 0

        while True:
            key = cv.waitKey(10)
            if key == 27:  # ESC key
                break

            number, mode = select_mode(key, mode)

            ret, image = cap.read()
            if not ret:
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks is not None:

                # Extract Keypoints
                no_of_hands = len(results.multi_hand_landmarks)
                pre_processed = []

                if no_of_hands == 2:
                    for hand in results.multi_hand_landmarks:
                        lm = calc_landmark_list(debug_image, hand)
                        pre_processed += pre_process_landmark(lm)
                        mp.solutions.drawing_utils.draw_landmarks(
                            debug_image, hand, mp_hands.HAND_CONNECTIONS)

                elif no_of_hands == 1:
                    pre_processed = [0.0] * 42
                    hand = results.multi_hand_landmarks[0]
                    lm = calc_landmark_list(debug_image, hand)
                    pre_processed += pre_process_landmark(lm)
                    mp.solutions.drawing_utils.draw_landmarks(
                        debug_image, hand, mp_hands.HAND_CONNECTIONS)

                else:
                    pre_processed = [0.0] * 84

                times_captured = logging_csv(number, mode, pre_processed, times_captured)

                hand_sign_id, confidence = keypoint_classifier(pre_processed)
                sign = keypoint_classifier_labels[hand_sign_id]

                # ---------------- SAFETY LOGIC ----------------
                # Do NOT exit when two hands are shown
                disable_exit = (no_of_hands == 2)

                # ---------------- FULL STOP EXIT (fixed) ----------------
                if sign == "fullstop" and confidence >= 0.85 and not disable_exit:
                    print("FULLSTOP detected — exiting...")
                    root.destroy()
                    break

                # Normal character prediction
                if confidence >= 0.85:
                    spcount = 0
                    if prevchar == sign:
                        count += 1
                    else:
                        prevchar = sign
                        count = 0

                    if count >= 10 and expression[-1] != sign:
                        prevchar = ' '
                        count = 0
                        enter_text(sign)

                debug_image = draw_info_text(debug_image, args.width, confidence, sign)

            else:
                spcount += 1
                if spcount >= 10 and expression[-1] != " ":
                    enter_text(" ")
                    spcount = 0

            # Display Frame
            debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            debug_image = draw_info(debug_image, mode, number, times_captured)

            img = ImageTk.PhotoImage(Image.fromarray(debug_image))
            l1['image'] = img
            root.update()

        cap.release()
        cv.destroyAllWindows()

    except Exception as e:
        print("Error:", e)


# ---------------- Utility Functions ----------------
def select_mode(key, mode):
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        number = len(list(csv.reader(f))) - 1

    if key == ord('n'):
        mode = 0

    if key == ord('k'):
        mode = 1
        sign_name = input("Enter sign name: ")
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  mode="a", newline="", encoding='utf-8-sig') as f:
            csv.writer(f).writerow([sign_name])

    return number, mode


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    points = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        points.append([x, y])
    return points


def pre_process_landmark(points):
    temp = copy.deepcopy(points)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y

    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs, temp)) or 1
    return [n / max_value for n in temp]


def logging_csv(number, mode, landmarks, count):
    if mode == 1:
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmarks])
        count += 1
    return count


def draw_info_text(image, w, confidence, text):
    if text != "" and confidence >= 0.7:
        cv.putText(image, text, (int(w / 2) + 10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return image


def draw_info(image, mode, number, captured):
    if mode == 1:
        cv.putText(image, "Reading Your Hand...", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv.putText(image, f"Label: {number}", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv.putText(image, f"Captured: {captured}", (10, 75),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    else:
        cv.putText(image, "Predicting...", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return image


if __name__ == '__main__':
    main()
