import pandas as pd
import tensorflow as tf

# Ruta de la carpeta que contiene los archivos Parquet distribuidos
carpeta = '.'

# Cargar la tabla distribuida desde los archivos Parquet
# Leer los archivos .parquet
df1 = pd.read_parquet("105143404.parquet")
df2 = pd.read_parquet("128822441.parquet")

# Seleccionar las columnas que no contienen 'face' o 'pose'
columns_to_delete_df = [col for col in df1.columns if 'face' in col.lower() or 'pose' in col.lower()]

# Eliminar las columnas excluidas del DataFrame original
df1 = df1.drop(columns=columns_to_delete_df)
df2 = df2.drop(columns=columns_to_delete_df)


# Concatenar los DataFrames
df_concatenado = pd.concat([df1, df2])

# Imprimir el DataFrame resultante
#print(df_concatenado.tail())

df_concatenado.to_csv("kk.csv", index=True)

# Combinar los DataFrames en uno solo

'''
# Acceder a un dato específico en el DataFrame
fila = 0  # Índice de la fila
columna = "frame"  # Nombre de la columna

valor = df[columna][video_id_deseado]


# Cargar el archivo CSV
df_labels = pd.read_csv("train.csv")
columns_to_delete_labels = [col for col in df_labels.columns if col != 'sequence_id' and col != 'phrase']
labels = df_labels.drop(columns=columns_to_delete_labels) 


# merge de los dos DataFrames
merged_df = df_reduc.merge(labels, left_index=True, right_on="sequence_id")

# Obtener los índices únicos
unique_indices = merged_df.index.unique()

# Barajar los índices únicos
shuffled_indices = tf.random.shuffle(unique_indices)

# Obtener el número total de índices únicos
num_indices = tf.shape(unique_indices)[0]

# Obtener el número de índices para cada conjunto
num_train = tf.cast(tf.cast(num_indices, tf.float32) * 0.7, tf.int32)
num_val = tf.cast(tf.cast(num_indices, tf.float32) * 0.15, tf.int32)
num_test = num_indices - num_train - num_val

# Dividir los índices únicos en conjuntos de entrenamiento, validación y prueba
train_indices = shuffled_indices[:num_train]
val_indices = shuffled_indices[num_train:num_train + num_val]
test_indices = shuffled_indices[num_train + num_val:]

# Obtener los conjuntos de datos correspondientes a los índices
train_df = merged_df.loc[merged_df.index.isin(train_indices)]
val_df = merged_df.loc[merged_df.index.isin(val_indices)]
test_df = merged_df.loc[merged_df.index.isin(test_indices)]

# Verificar las formas de los conjuntos de datos resultantes
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# Guardar los conjuntos de datos en archivos CSV
train_df.to_csv("net_train.csv", index=False)
val_df.to_csv("net_val.csv", index=False)
test_df.to_csv("net_test.csv", index=False)
'''