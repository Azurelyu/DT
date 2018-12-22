from Dession_Tree import DessionTree
import random as rd
import numpy as np
import json


class random_forest:
    def __init__(self,max_trees=6,max_deep=6,cate='val'):
        self.max_trees=max_trees   #最多多少树
        self.max_deep=max_deep    #树的最大深度
        self.cate=cate   #决策树的类型
        self.trees_dic={}
        for i in range(self.max_trees):
            self.trees_dic[i]=DessionTree(max_deep=max_deep,cate=cate)

    def dis_the_data(self,X,Y,k):
        out_X=[]
        out_Y=[]
        idx_list=[]
        for i in range(k):
            rd_=rd.randint(0,len(X)-1)
            if rd_ not in idx_list:
                idx_list.append(rd_)

        for i in idx_list:
            out_X.append(X[i])
            out_Y.append(Y[i])
        return np.array(out_X),np.array(out_Y)

    def fit(self,X,Y):      #训练的过程
        for i in range(self.max_trees):
            print('training NO.'+str(i)+' tree')
            X_tmp,Y_tmp=self.dis_the_data(X,Y,int(len(X)/self.max_trees))
            self.trees_dic[i].fit(X_tmp,Y_tmp)

    def predict(self,X):
        y_list=[]
        for i in range(self.max_trees):
            y_list.append(self.trees_dic[i].predict(X))
        return np.mean(np.array(y_list),0)

    def save_model(self, modelname):
        with open(modelname, 'w', encoding='utf-8') as fo:
            json.dump(self.get_dic(), fo, ensure_ascii=False)

    def load_model(self, modelname):
        with open(modelname, 'r', encoding='utf-8') as fo:
            tmp_dic = json.load(fo)
        for key in tmp_dic:
            dt = DessionTree()
            dt.load_model_by_dic(tmp_dic[key])
            self.trees_dic[int(key)] = dt

        self.max_trees = len(tmp_dic)

    def get_dic(self):
        out_dic = {}
        for key in self.trees_dic:
            out_dic[key] = self.trees_dic[key].tree.get_dic()
        return out_dic

if __name__ == '__main__':

    X = np.load('X.npy')
    Y = np.load('Y.npy')
    x_test = X[:1000]
    y_test = Y[:1000]
    x_train = X[1000:]
    y_train = Y[1000:]
    import sys
    import os

    cate = 'val'
    if cate != 'val' and cate != 'id3' and cate != 'c4.4':
        print('cate error')
    else:
        if not os.path.exists('Random_forest_model.' + cate) or (len(sys.argv)==3 and sys.argv[2]=='force'):
            dd = random_forest(max_trees=6, max_deep=6, cate=cate)
            dd.fit(x_train, y_train)
            dd.save_model('Random_forest_model.' + cate)

        dd1 = random_forest()
        dd1.load_model('Random_forest_model.' + cate)

        y = dd1.predict(x_test)
        for idx, i in enumerate(y):
            print(i, y_test[idx])
        print(np.mean((y - y_test) * (y - y_test)))




