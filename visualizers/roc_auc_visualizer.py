import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def roc_curve_visualizer(y_true_record, y_score_record, img_name):

    img_path = './imgs/' + img_name + '_roc_curve'

    y_true, y_score = y_true_record, y_score_record

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()

    plt.subplot(122)
    plt.plot(thresholds, fpr)
    plt.plot(thresholds, tpr)
    plt.xlim(0, 1)
    plt.xlabel('Threshold of score')
    plt.legend(['False positive rate', 'True positive rate'])
    plt.grid()

    plt.savefig(img_path + '.jpg')

    plt.clf()