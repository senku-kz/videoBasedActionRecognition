import cv2
import numpy as np
from matplotlib import pyplot as plt

from keypoints import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from dataset import actions
from model import LstmNeuralNetwork

from video_file_play import urls

colors = [
    (245, 117, 16), (117, 245, 16), (200, 117, 100), (16, 117, 245), (150, 20, 245),
    (245, 117, 16), (117, 245, 16), (200, 117, 100), (16, 117, 245), (150, 20, 245),
          ]


def prob_viz(res, actions, input_frame, colors):
    h_rec = 20
    h_txt = h_rec + 25
    text_color_bgr = (255, 255, 255)

    output_frame = input_frame.copy()

    for num, prob in enumerate(res):

        cv2.rectangle(output_frame, (0, h_rec + num * 40), (int(prob * 100), (h_rec+30) + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, h_txt + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    text_color_bgr, 2, cv2.LINE_AA)

    return output_frame


if __name__ == '__main__':
    model = LstmNeuralNetwork()
    model.model_load()

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    for video_file_url in urls:

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

                    # Draw landmarks
                    draw_styled_landmarks(image, results)

                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.model_predict(np.expand_dims(sequence, axis=0))[0]
                        class_idx = np.argmax(res)
                        print(actions[class_idx])

                        # 3. Viz logic
                        if res[class_idx] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 3:
                            sentence = sentence[-3:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image, colors)

                    # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (20, 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (99, 49, 222), 2, cv2.LINE_AA)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                else:
                    break

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            capture.release()
            cv2.destroyAllWindows()
