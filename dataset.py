import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keypoints import mediapipe_detection, mp_holistic, draw_styled_landmarks, extract_keypoints

# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('ucf_sports_actions/ucf action')

actions = [
    "Diving",
    "Golf-Swing",
    "Kicking",
    "Lifting",
    "Riding-Horse",
    "Running",
    "SkateBoarding",
    "Swing-Bench",
    "Swing-Side",
    "Walking",
]

directory_action_mapper = {
    "Diving-Side": "Diving",
    "Golf-Swing-Back": "Golf-Swing",
    "Golf-Swing-Front": "Golf-Swing",
    "Golf-Swing-Side": "Golf-Swing",
    "Kicking-Front": "Kicking",
    "Kicking-Side": "Kicking",
    "Lifting": "Lifting",
    "Riding-Horse": "Riding-Horse",
    "Run-Side": "Running",
    "SkateBoarding-Front": "SkateBoarding",
    "Swing-Bench": "Swing-Bench",
    "Swing-SideAngle": "Swing-Side",
    "Walk-Front": "Walking",
}


class ActionDataset:
    def __init__(self):
        self.directory_root = "ucf_sports_actions/ucf action"
        self.directory_list = None
        self.video_file_url = None

        self.sequence_length = 20  # Videos are going to be 30 frames in length
        self.dataset_action = "dataset_action"

        self.label_map = {label: num for num, label in enumerate(actions)}

    def read_folders_list(self):
        self.directory_list = [
            directory_class for directory_class in os.listdir(
                self.directory_root) if os.path.isdir(os.path.join(self.directory_root, directory_class))
        ]
        self.directory_list.sort()
        print(self.directory_list)

    def read_files_list(self):
        self.video_file_url = []

        for directory_class in self.directory_list:
            url1 = os.path.join(self.directory_root, directory_class)
            sq = os.listdir(url1)
            sq.sort()
            for directory_sample in sq:
                url2 = os.path.join(self.directory_root, directory_class, directory_sample)
                if os.path.isdir(url2):
                    for f in os.listdir(url2):
                        url3 = os.path.join(self.directory_root, directory_class, directory_sample, f)
                        if f.endswith('.avi'):
                            self.video_file_url.append((directory_class, url3))

    def gen_dataset_img(self):
        if not os.path.exists(self.dataset_action):
            os.makedirs(self.dataset_action)

        sample = 0
        for dir_class, video_file_url in self.video_file_url:
            print(directory_action_mapper[dir_class], video_file_url)

            url = os.path.join(self.dataset_action, directory_action_mapper[dir_class])
            if not os.path.exists(os.path.join(url)):
                os.makedirs(os.path.join(url))

            capture = cv2.VideoCapture(filename=video_file_url)

            count = 0
            while capture.isOpened():

                # Read feed
                success, frame = capture.read()

                if success:
                    dir_url = os.path.join(url, "{:03d}".format(sample))
                    if not os.path.exists(dir_url):
                        os.makedirs(dir_url)

                    url_file = os.path.join(dir_url, "frame_{:03d}.jpg")

                    cv2.imwrite(url_file.format(count), frame)  # save frame as JPEG file

                    # Show to screen
                    cv2.imshow('OpenCV Feed {}'.format(directory_action_mapper[dir_class]), frame)

                    count += 1
                    if count >= self.sequence_length:
                        sample += 1
                        count = 0
                        # return
                else:
                    break

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            capture.release()
            cv2.destroyAllWindows()

    def gen_dataset_np(self):
        if not os.path.exists(self.dataset_action):
            os.makedirs(self.dataset_action)

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            sequence = 0
            for dir_class, video_file_url in self.video_file_url:
                print(directory_action_mapper[dir_class], video_file_url)

                url = os.path.join(self.dataset_action, directory_action_mapper[dir_class])

                if not os.path.exists(os.path.join(url)):
                    os.makedirs(os.path.join(url))

                capture = cv2.VideoCapture(filename=video_file_url)

                count = 0

                if not capture.isOpened():
                    print("Error opening video stream or file")

                while capture.isOpened():

                    # Read feed
                    success, frame = capture.read()

                    if success:
                        dir_url = os.path.join(url, "{:03d}".format(sequence))
                        if not os.path.exists(dir_url):
                            os.makedirs(dir_url)

                        try:
                            # Make detections
                            image, results = mediapipe_detection(frame, holistic)
                        except:
                            continue

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        # Export keypoints
                        keypoints = extract_keypoints(results)

                        url_img_file = os.path.join(dir_url, "frame_{:03d}.jpg".format(count))
                        cv2.imwrite(url_img_file, image)  # save image as JPEG file
                        # cv2.imwrite(url_img_file, frame)  # save frame as JPEG file

                        url_npy_file = os.path.join(dir_url, "frame_{:03d}".format(count))
                        np.save(url_npy_file, keypoints)
                        # print(url_npy_file)

                        # Show to screen
                        cv2.imshow('OpenCV Feed {}'.format(directory_action_mapper[dir_class]), image)

                        count += 1
                        if count >= self.sequence_length:
                            sequence += 1
                            count = 0

                    else:
                        break

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # sequence += 1
                capture.release()
                cv2.destroyAllWindows()

    def process_data(self):
        print("label map", self.label_map)
        sequences, labels = [], []

        for action in actions:
            url1 = os.path.join(self.dataset_action, action)
            sequence_lst = os.listdir(url1)
            for sequence_dir in sequence_lst:
                if os.path.isdir(os.path.join(url1, sequence_dir)):
                    file_lst = os.listdir(os.path.join(url1, sequence_dir))
                    file_lst.sort()
                    window = []
                    for np_filename in file_lst:
                        if np_filename.endswith(".npy"):
                            print(os.path.join(url1, sequence_dir, np_filename))
                            res = np.load(os.path.join(url1, sequence_dir, np_filename))
                            if res.any():
                                window.append(res)
                        else:
                            continue

                    s = 0
                    n = len(window)
                    while n >= self.sequence_length:
                        sequences.append(window[s:s+self.sequence_length])
                        labels.append(self.label_map[action])
                        s += self.sequence_length
                        n -= self.sequence_length

        print(np.array(sequences).shape)  # window_count, window_size, pose_keypoints
        print(np.array(labels).shape)

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        print("X.shape", X.shape)
        print("y.shape", y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)
        print("y_test", y_test.shape)

        np.save(os.path.join(self.dataset_action, "X_train.npy"), X_train)
        np.save(os.path.join(self.dataset_action, "y_train.npy"), y_train)
        np.save(os.path.join(self.dataset_action, "X_test.npy"), X_test)
        np.save(os.path.join(self.dataset_action, "y_test.npy"), y_test)
        print("Train and Test dataset saved successfully!")


if __name__ == '__main__':
    print(actions)

    obj = ActionDataset()
    obj.read_folders_list()
    obj.read_files_list()
    obj.gen_dataset_np()
    obj.process_data()
