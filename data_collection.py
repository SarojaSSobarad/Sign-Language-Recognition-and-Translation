import cv2
import csv
import mediapipe as mp
from sign_gesture import calc_landmark_list, pre_process_landmark  # uses your existing functions

CSV_PATH = "model/keypoint_classifier/keypoint.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)

    print("\n========== DATASET COLLECTION ==========\n")
    label = input("Enter the gesture you want to collect: ")
    print("\nInstructions:")
    print("➡ Show your gesture in front of the camera")
    print("➡ Press 'k' to SAVE one sample")
    print("➡ Press 'n' to QUIT\n")

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            # Extract landmarks
            lm = calc_landmark_list(frame, hand)

            # Preprocess (normalize + flatten)
            pre = pre_process_landmark(lm)

            # Draw landmarks on screen
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Show window
        cv2.putText(frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Saved: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Dataset Collector", frame)

        key = cv2.waitKey(1)

        # Save sample
        if key == ord('k') and results.multi_hand_landmarks:
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([label] + pre)
            count += 1
            print(f"Sample {count} saved")

        # Quit
        if key == ord('n'):
            print("\nExiting dataset collection...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
