import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fm


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = fm.FaceDetector()

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        succes, img = cap.read()
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        # calculate frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()