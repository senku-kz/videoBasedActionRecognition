import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from dataset import actions
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


class LstmNeuralNetwork:
    def __init__(self):
        self.dataset_action = "dataset_action"
        self.X_train = np.load(os.path.join(self.dataset_action, "X_train.npy"))
        self.y_train = np.load(os.path.join(self.dataset_action, "y_train.npy"))
        self.X_test = np.load(os.path.join(self.dataset_action, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.dataset_action, "y_test.npy"))
        self.bath_size = 128
        self.epochs = 500

        self.model = None

        log_dir = os.path.join('Logs')
        self.tb_callback = TensorBoard(log_dir=log_dir)

        self.define_the_model()
        self.compiling_the_model()

    def define_the_model(self):
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 33*4)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(actions), activation='softmax'))

    def compiling_the_model(self):
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def train_the_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs,
                       batch_size=self.bath_size,
                       callbacks=[self.tb_callback]
                       )

    def model_summary(self):
        self.model.summary()

    def model_evaluation(self):
        # res = self.model.predict(self.X_test)
        # actions_predicted = actions[np.argmax(res)]

        yhat = self.model.predict(self.X_test)
        ytrue = np.argmax(self.y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        multilabel_confusion_matrix(ytrue, yhat)
        acc_score = accuracy_score(ytrue, yhat)
        print("Test dataset size:", self.X_test.size)
        print("Accuracy score:", acc_score)

    def model_predict(self, image):
        return self.model.predict(image)

    def model_save(self):
        # self.model.save('action.h5')
        self.model.save('action.keras')
        print("Model saved successfully")

    def model_load(self):
        self.model.load_weights('action.keras')
        print("Model loaded successfully")


if __name__ == '__main__':
    obj = LstmNeuralNetwork()
    # obj.train_the_model()
    # obj.model_summary()
    # obj.model_save()
    obj.model_load()
    obj.model_evaluation()
