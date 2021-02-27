import pickle
from core.evaluation import evaluate

from core.config import get_config


if __name__ == '__main__':
    cfg = get_config()

    with open(cfg.WORK_DIR + 'predictions.json', 'rb') as f:
        preds = pickle.load(f)
    preds = preds['predictions']

    name_value = evaluate(cfg, preds)
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean', 'Mean@0.1'))
    print('|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|%-10s|' % ('%.3f' % name_value['Head'],
                                                                       '%.3f' % name_value['Shoulder'],
                                                                       '%.3f' % name_value['Elbow'],
                                                                       '%.3f' % name_value['Wrist'],
                                                                       '%.3f' % name_value['Hip'],
                                                                       '%.3f' % name_value['Knee'],
                                                                       '%.3f' % name_value['Ankle'],
                                                                       '%.3f' % name_value['Mean'],
                                                                       '%.3f' % name_value['Mean@0.1']))

