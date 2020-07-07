from featureExtract import extract
from utils import save_dir

if __name__ == '__main__':

    features, labels = extract(input='../data/net/train/',
                               if_predict=False,
                               if_train=True,
                               paramPath='featureExtract/epoch__26_acc_0.85net_param.pkl')

    save_dir(features, 'X_train', '../data/trainFeature')
    save_dir(labels, 'Y_train', '../data/trainFeature')
