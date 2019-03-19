import os
import pandas as pd

data_root = '../data/mozilla_common_voice/eo'
datasets = {
    'train.tsv': 'filelists/mcv_eo_train_filelist.txt',
    'test.tsv': 'filelists/mcv_eo_test_filelist.txt',
    'dev.tsv': 'filelists/mcv_eo_val_filelist.txt'
}

for src, dest in datasets.items():
    src = os.path.join(data_root, src)
    data = pd.read_csv(src, sep='\t')

    with open(dest, 'w') as fl:
        for fname, text in zip(data.path, data.sentence):
            fl.write(
                f'{data_root}/clips/{fname}.mp3|{text}\n'
                )
