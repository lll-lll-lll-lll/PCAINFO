"""主成分分析とは
説明変数の中に相関関係にある複数の変数が存在すると、モデルはその影響を強く受けバイアスがかかる。
これを解消するのが主成分分析
<用語>
寄与率: 各変数がどれくらい重要か。各主成分で説明できるデータの割合を示す
因⼦負荷量: 各変数の各主成分への影響⼒ → 各主成分の意味を推定できる
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
from sklearn.decomposition import PCA
from pandas import plotting 


class PCAINFO(PCA):
    def __init__(self, data:pd.DataFrame):
        super().__init__()
        self.data = data
        self.pca = super().fit(data)
        self.feature = super().transform(self.data) # データを主成分空間に写像
        self.columns = ["PC{}".format(x + 1) for x in range(len(self.data.columns))]
        self.df = self.get_pca_pd()
        self.eig = self.get_eig() # 固有値
        self.conb_rate = self.get_conb_rate() # 寄与率
        self.total_conb_rate = self.get_total_conb_rate()
        

    def get_pca_pd(self)->pd.DataFrame:
         # 主成分得点
         df = pd.DataFrame(self.feature, columns=self.columns)
         return df
    
    def test(self):
        return self.pca.components_

    def plt_pca(self, feature_column=[0,1]):
        # デフォルト：第一主成分と第二主成分でプロットする
        # feature_column：成分を選ぶ
        plt.figure(figsize=(6, 6))
        plt.scatter(self.feature[:, feature_column[0]], self.feature[:, feature_column[1]], alpha=0.8, c=list(self.df.iloc[:, 0]))
        plt.grid()
        plt.xlabel(f'PC{feature_column[0]}')
        plt.ylabel(f'PC{feature_column[1]}')
        plt.show()
    
    def plt_contribution(self, comp=[0,1]):
        # デフォルト第一主成分と第二主成分における観測変数の寄与度をプロット
        plt.figure(figsize=(6, 6))
        for x, y, name in zip(self.pca.components_[comp[0]], self.pca.components_[comp[1]], self.data.columns[1:]):
            plt.text(x, y, name)
        plt.scatter(self.pca.components_[comp[0]], self.pca.components_[comp[1]], alpha=0.8)
        plt.grid()
        plt.xlabel(f'PC{comp[0]}')
        plt.ylabel(f'PC{comp[1]}')
        plt.show()
        
    

    def plt_matrix(self):
        plotting.scatter_matrix(pd.DataFrame(self.feature, 
                        columns=self.columns), 
                        figsize=(8, 8), c=list(self.df.iloc[:, 0]), alpha=0.5)
        plt.show()
    

    def get_eigenvec(self):
        # 各主成分の固有ベクトル
        eig_vec = pd.DataFrame(self.pca.components_, index = self.pca.columns,
                          columns = self.columns)
        return eig_vec

    
    def get_eig(self):
        # 固有値を得る
        eig = pd.DataFrame(self.pca.explained_variance_, index=self.columns, columns=['固有値']).T
        return eig
    
    def get_conb_rate(self):
        # 寄与率を得る
        conb_rate = pd.DataFrame(self.pca.explained_variance_ratio_, index=self.columns, columns=['寄与率']).T
        return conb_rate

    def get_total_conb_rate(self):
        # 累積寄与率を得る
        total_conb_rate = pd.DataFrame(self.pca.explained_variance_ratio_.cumsum(), index=self.columns, columns=['累積寄与率']).T
        return total_conb_rate

    def get_std_eig(self):
        # 固有値の標準偏差を得る
        dv = np.sqrt(self.eig)
        dv = dv.rename(index = {'固有値':'主成分の標準偏差'})
        return dv

