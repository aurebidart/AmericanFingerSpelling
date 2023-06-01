import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model as m
import pandas as pd
import numpy as np

# Cargar los datos

import csv

mapeo = {
    " ": 0, "!": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10, ",": 11,
    "-": 12, ".": 13, "/": 14, "0": 15, "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22,
    "8": 23, "9": 24, ":": 25, ";": 26, "=": 27, "?": 28, "@": 29, "[": 30, "_": 31, "a": 32, "b": 33,
    "c": 34, "d": 35, "e": 36, "f": 37, "g": 38, "h": 39, "i": 40, "j": 41, "k": 42, "l": 43, "m": 44,
    "n": 45, "o": 46, "p": 47, "q": 48, "r": 49, "s": 50, "t": 51, "u": 52, "v": 53, "w": 54, "x": 55,
    "y": 56, "z": 57, "~": 58
}

# Leer el archivo CSV
lector_csv = pd.read_csv('train.csv', usecols=['phrase'])
y_train = []

print(lector_csv.values)
# Iterar sobre cada fila del archivo CSV
for fila in lector_csv.values:

    phrase = [0] * 59
    
    print(fila[0])
    for caracter in fila[0]:
        phrase[mapeo[caracter]] = 1

    y_train.append(phrase)

tensor = np.asarray(y_train).astype('float32')
print(tensor.shape)
np.save('y_train.npy', tensor)

'''
parquet_ids = [5414471, 105143404, 128822441, 149822653, 152029243, 169560558, 175396851, 234418913, 296317215]

total = []

for id in parquet_ids:
    df = pd.read_parquet(f'./{id}.parquet')

    columns_to_delete_df = [col for col in df.columns if 'face' in col.lower() or 'pose' in col.lower()]
    df = df.drop(columns=columns_to_delete_df)
    df.fillna(0)

    for i in df.index.unique():
        if i == 1818239060 or i == 201728475 or i == 1547755601 or i == 120620494:
            continue

        ar = []

        for frame in df.loc[i]['frame']:
            fila = df.loc[i].loc[df.loc[i]['frame'] == frame]
            columnas = fila.columns[1:]
            ar.append(fila.loc[:, columnas].values.flatten())

        total.append(ar)
        print(i)
    
    print(id)

tensor = tf.ragged.constant(total)
np.save('tensor_ragged_filled.npy', tensor)
print(tensor.shape)

tensor = tensor.to_tensor()
np.save('tensor_dense_filled.npy', tensor)
print(tensor.shape)
'''