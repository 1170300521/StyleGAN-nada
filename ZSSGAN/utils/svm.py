from turtle import pd
import numpy as np
from sklearn import svm
from torch import ne, neg, neg_


def train_boundary(pos_codes, neg_codes, split_ratio=0.7):
    pos_ids = np.arange(len(pos_codes))
    np.random.shuffle(pos_ids)
    train_pos_num = int(len(pos_ids) * split_ratio)
    train_pos_codes = pos_codes[pos_ids[:train_pos_num]]
    val_pos_codes = pos_codes[pos_ids[train_pos_num:]]

    neg_ids = np.arange(len(neg_codes))
    np.random.shuffle(neg_ids)
    train_neg_num = int(len(neg_ids) * split_ratio)
    train_neg_codes = neg_codes[neg_ids[:train_neg_num]]
    val_neg_codes = neg_codes[neg_ids[train_neg_num:]]

    train_data = np.concatenate([train_pos_codes, train_neg_codes], axis=0)
    train_label = np.concatenate([np.ones(train_pos_num, dtype=np.int),
                                    np.zeros(train_neg_num, dtype=np.int)], axis=0)
    print(f'Training: {train_pos_num} positive, {train_neg_num} negtive.')

    val_data = np.concatenate([val_pos_codes, val_neg_codes], axis=0)
    val_label = np.concatenate([np.ones(len(val_pos_codes)),
                                np.zeros(len(val_neg_codes))], axis=0)
    print(f'Validation: {len(val_pos_codes)} positive, {len(val_neg_codes)} negtive.')

    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)

    if len(val_label) > 0:
        val_pred = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_pred)
        print(f'Accurracy for validattion set: {correct_num} / {len(val_label)} = {correct_num / len(val_label):.6f}.')
    
    a = classifier.coef_.reshape(1, pos_codes.shape[1]).astype(np.float32)
    return a / np.linalg.norm(a)

def get_delta_w(pos_path, output_path, neg_path="/home/ybyb/CODE/StyleGAN-nada/results/invert/A_gen_w.npy"):
    pos_codes = np.load(pos_path).reshape((-1, 18, 512))[:, :]
    neg_codes = np.load(neg_path).reshape((-1, 18, 512))[:, :]
    chosen_num = min(10000, len(neg_codes))
    pos_num = min(500, len(pos_codes))
    # np.save("/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/mean_delta_w.npy", (pos_codes.mean(0) - neg_codes.mean(0)))
    np.random.shuffle(pos_codes)
    np.random.shuffle(neg_codes)
    pos_codes = pos_codes[0:pos_num].reshape((pos_num, -1))
    neg_codes = neg_codes[0:chosen_num].reshape((chosen_num, -1))
    a = train_boundary(pos_codes, neg_codes, split_ratio=0.7)
    np.save(output_path, a.reshape((-1, 512)))

if __name__ == "__main__":
    pos_path = "/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/B_codes.npy"
    # neg_path = "/home/ybyb/CODE/StyleGAN-nada/results/invert/ffhq_w+.npy"
    neg_path = "/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/A_codes.npy"
    output_path = "/home/ybyb/CODE/StyleGAN-nada/results/demo_ffhq/photo+Image_1/test/small_delta_w.npy"
    get_delta_w(pos_path, neg_path, output_path)