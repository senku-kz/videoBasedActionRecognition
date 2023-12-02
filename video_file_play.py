import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose_keypoints


def play_video_file(video_file_url):
    capture = cv2.VideoCapture(filename=video_file_url)

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while capture.isOpened():

            # Read feed
            success, frame = capture.read()

            if success:

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                keypoints = extract_keypoints(results)
                # print(result_test, result_test.shape)
                # np.save('0', result_test)
                # np.load('0.npy')

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)  # Draw pose connections

                # draw_landmarks(image, results)
                # draw_styled_landmarks(image, results)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

            else:
                break

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # capture.release()
        # cv2.destroyAllWindows()


urls = [
    "ucf_sports_actions/ucf action/Golf-Swing-Side/001/RF1-13207_7015.avi",
    "ucf_sports_actions/ucf action/Lifting/001/3528-8_70514.avi",
    "ucf_sports_actions/ucf action/Riding-Horse/002/4456-16_700165.avi",
    "ucf_sports_actions/ucf action/Run-Side/0055238-17_701581.avi",
    "ucf action/SkateBoarding-Front/009/947-58108.avi",
    "ucf_sports_actions/ucf action/Walk-Front/006/RF1-13902_70016.avi",
    "ucf_sports_actions/ucf action/Walk-Front/013/RF1-18524_70031.avi",
]


if __name__ == '__main__':
    for url in urls:
        play_video_file(url)
