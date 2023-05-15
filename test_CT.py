#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

import numpy as np

def seg_eval(y_pred, y_true):

    # y_true = y_true[0]
    # print(y_true.shape)
    # print(y_true.shape)
    # print(y_pred.shape)
    y_pred = np.argmax(y_pred,0)
    classes=4
    y_pred = np.eye(classes)[y_pred]

    f1 = f1_score(y_true, y_pred, average='macro')
    reall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')




    return f1, reall, precision





def test(data_loader, model,  sets):
    f1_all = []
    reall_all = []
    precision_all = []

    model.eval()  # for testing
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volumes, label = batch_data  # [1, 1, 28, 224, 224], [1]  前面是输入的nii.gz特征，后面是label
        if not sets.no_cuda:
            volumes = volumes.cuda()
            label = label.cuda().long()
        out = model(volumes)
        # pred = torch.argmax(out, 1)
        f1, reall, precision = seg_eval(out[0], label[0])
        f1_all.append(f1)
        reall_all.append(reall)
        precision_all.append(precision)


    return np.mean(f1_all), np.mean(reall_all), np.mean(precision_all)


# In[ ]:




