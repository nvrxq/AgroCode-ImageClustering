import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm.auto import tqdm
from skimage.transform import resize
import random
from utils import CFG as utils





class _paths:
    _queries = pd.read_csv('data/queries.csv')

    _test = pd.read_csv('data/test.csv')
    test_path = [Path('data/test/{}.png'.format(id)) for id in _test.idx]
    queries_path = [Path('data/queries/{}.png'.format(id)) for id in _queries.idx]



class ImgData(object):
    def __init__(self, path=None):
        self.path = path

    def read_image(self, path: str) -> tf.Tensor:
        '''
        Channels : 3
        ------------------
        1.Reading File
        2. Decode([6] photo have 4 channels)
        3. Change type to tf.float32
        4. (3, 512, 512) -> (1, 3, 512, 512) ()
        '''
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        return img

    def _embFeatures(self, model) -> np.ndarray:
        '''
        from_tensor_slices : make dataset dimension
        ParallelMapDataset : creates a dataset to apply

        '''
        _files = list(map(str, self.path))
        _files = tf.data.Dataset.from_tensor_slices(_files)
        _files = _files.map(self.read_image,
                            num_parallel_calls=tf.data.AUTOTUNE)
        _files = _files.map(utils.resize_images,
                            num_parallel_calls=tf.data.AUTOTUNE)  # resize to 512x512
        _files = _files.prefetch(10)
        # get embeddings & l2 norm
        _emb = _files.map(model)
        _emb = _emb.map(utils._l2)

        _total = len(_emb)
        _emb = _emb.as_numpy_iterator()
        _emb = tqdm(_emb, total=_total)
        _emb = list(_emb)
        _emb = np.concatenate(_emb)

        return _emb


model = tf.keras.models.load_model('./weights/')


test_images = ImgData(_paths.test_path)
queries_images = ImgData(_paths.queries_path)

test_emb = test_images._embFeatures(model)
queries_emb = queries_images._embFeatures(model)


def find_similar(queries_emb,test_emb,count_predict):
    scores = utils.cosine_distance(queries_emb,test_emb)
    nq, ndb = scores.shape
    sorted_scores = scores.argsort(axis=1)[:, ::-1]  # sort descending per row
    topk = sorted_scores[:, : count_predict]  # get **indices** of the topk images for each row

    topk_scores = np.stack([scores[i, topk[i]] for i in range(nq)])  # get topk scores (comprehensible version)
    return topk, count_predict, topk_scores


topk, k, topk_scores= find_similar(queries_emb,test_emb,10)


predicted = []
batch = []
for item in topk:
    for index in item:
        batch.append(_paths._test.idx.iloc[index])
    predicted.append(batch)
    batch = []
predicted = np.array(predicted)

pred_data = pd.DataFrame()
pred_data['score'] = topk_scores.flatten()
pred_data.loc[:, 'query_idx'] = np.repeat(_paths._queries.idx, 10).values
pred_data['database_idx'] = predicted.flatten()
pred_data.to_csv('data/submission.csv', index=False)
