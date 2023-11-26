# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from dataset import ActionDataset
from model import LstmNeuralNetwork

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ds = ActionDataset()
    ds.read_folders_list()
    ds.read_files_list()
    ds.gen_dataset_np()
    ds.process_data()

    nn = LstmNeuralNetwork()
    nn.train_the_model()
    nn.model_summary()
    nn.model_save()
    nn.model_load()
    nn.model_evaluation()

    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
