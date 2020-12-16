import csv
import random
import math
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


data = pd.read_csv(r'F:\CSI_Indoor_Localization\fingerprint_no3n5.csv')

x_train = data[{'1','2','3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','25','26','27','28','29','30',
                '31','32','33','34','35','36','37','38','39','40',
                '41','42','43','44','45','46','47','48','49','50',
                '51','52','53','54','55','56','57','58','59','60'}]

y_train = data['61']

data2 = pd.read_csv(r'F:\CSI_Indoor_Localization\fingerprint_3n5_test.csv')

x_test = data2[{'1','2','3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','25','26','27','28','29','30',
                '31','32','33','34','35','36','37','38','39','40',
                '41','42','43','44','45','46','47','48','49','50',
                '51','52','53','54','55','56','57','58','59','60'}]

y_test = data2['61']

mlp = MLPClassifier(hidden_layer_sizes=3, activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001, max_iter=300)
mlp.fit(x_train, y_train)
print('MLP accuracy with 3 hidden layers:',mlp.score(x_test, y_test))

mlp = MLPClassifier(hidden_layer_sizes=5, activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001, max_iter=300)
mlp.fit(x_train, y_train)
print('MLP accuracy with 5 hidden layers:',mlp.score(x_test, y_test))

mlp = MLPClassifier(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001, max_iter=300)
mlp.fit(x_train, y_train)
print('MLP accuracy with 100 hidden layers:',mlp.score(x_test, y_test))