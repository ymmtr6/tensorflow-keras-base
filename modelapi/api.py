# coding:utf-8
import sys
import os
import glob

sys.path.append("../")

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import *
import dataprocesser.gmn as gmn

"""

"""
class GMNAPI(object):

    def __init__(self):
        """
        初期化
        """
        self.model = None
        self.x = None

    def setModel(self, model_filepath):
        """
        学習済みモデル(hoge.h5)
        """
        self.model = load_model(model_filepath)

    def predict_by_dir(self, filepath):
        """
        filepathで指定したディレクトリの中に存在してるjpgを全て判定する．
        """
        x = self.load_data(filepath)
        result = self.model.predict(x)
        print(result)

    def load_data(self, filepath):
        """
        データの読み込み部, gmn(dataprocesser)に移譲する
        """
        x = gmn.load_pred_data(filepath)
        x = np.array(x)
        return x

"""

"""
if __name__ = "__main__":
    args = sys.argv
    if len(args) == 1:
        print("[USAGE]: python gmn_api.py [model path] [pic dir]")
        exit()
    api = GMNAPI()
    api.setModel(args[1])
    api.predict_by_dir(args[2])
