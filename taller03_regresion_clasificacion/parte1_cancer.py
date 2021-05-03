#%% [markdown]  
"""
# Taller Regresión y clasificación - Parte 1
## Autor: Samuel Mesa
## Fecha: 20 de abril de 2021

## Objetivo: Construya una red neuronal de clasificación binaria para predecir el cáncer de seno.  Los datos los puede encontrar en el enlace de Kaggle.

Importar las librerias para la lectura de datos, gráficas y para definir el modelo de los datos
"""
#%% [markdown]
"""
### Meta-información sobre los datos
* ID number
* Diagnosis (M = maligno, B = benigno)

Se calculan diez características de valor real para cada núcleo celular:

* radius (media de las distancias desde el centro a los puntos del perímetro)
* texture (desviación estándar de los valores de la escala de grises)
* perimeter
* area
* smoothness (variación local en las longitudes de los radios)
* compactness (perímetro ^ 2 / área - 1.0)
* concavity (severidad de las porciones cóncavas del contorno)
* concave points (número de porciones cóncavas del contorno)
* symetry
* fractal dimension ("aproximación de la línea de costa"

La media, el error estándar y el "peor" o el mayor (la media de los tres valores más grandes) de estas características se calcularon para cada imagen, lo que resultó en 30 características. Por ejemplo, el campo 3 es Radio medio, el campo 13 es Radio SE, el campo 23 es Peor radio.

Todos los valores de las características se recodifican con cuatro dígitos significativos.

datos faltantes: ninguno

Distribución de clases: 357 benignos, 212 malignos
"""
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.estimator import LinearClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.utils import plot_model
#%% [markdown]  
# Cargar los datos desde el CSV local y describir la fuente de datos. Los datos corresponden
#%%
#df = pd.read_csv('BreastCancerWisconsin.csv', index_col=0)
df = pd.read_csv('https://raw.githubusercontent.com/DiplomadoAI-DL-UNAL2021/samuelmesa/main/taller03_regresion_clasificacion/BreastCancerWisconsin.csv', index_col=0)
del df['Unnamed: 32']
df.head()
#%% [markdown]
# # Preparar los datos
# Se realiza la preperación de los datos 
#%%
df_X = df.iloc[:,1:]
df_Y = df.apply(lambda x: 1 if x['diagnosis'] == 'M' else 0, axis=1)
#%% Describir los datos iniciales

df_describe = df.describe().transpose()
print(df_describe)
correlation_data = df_X.corr()
correlation_data.style.background_gradient(cmap='coolwarm', axis=None)  
#%%  Preparamos los datos de entrenamiento y testeo

training_features , test_features ,training_labels, test_labels = train_test_split(df_X , df_Y , test_size=0.2)
print('No. of rows in Training Features: ', training_features.shape[0])
print('No. of rows in Test Features: ', test_features.shape[0])
print('No. of columns in Training Features: ', training_features.shape[0])
print('No. of columns in Test Features: ', test_features.shape[0])

print('No. of rows in Training Label: ', training_labels.shape[0])
print('No. of rows in Test Label: ', test_labels.shape[0])
print('No. of columns in Training Label: ', training_labels.shape[0])
print('No. of columns in Test Label: ', test_labels.shape[0])
stats = training_features.describe()
stats = stats.transpose()
stats
stats = test_features.describe()
stats = stats.transpose()
#%% Normalización de datos

def norm(x):
  stats = x.describe()
  stats = stats.transpose()
  return (x - stats['mean']) / stats['std']

normed_train_features = norm(training_features)
normed_test_features = norm(test_features)
#%% Defina el generador de entrada de datos

def get_input_fn(data_training, data_labels): 
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_training), data_labels))\
        .shuffle(True).batch(32).repeat(1)
    return dataset

#%% Modelo ANN
train_fn_feeds = lambda: get_input_fn(normed_train_features,training_labels)

feature_columns_numeric = [tf.feature_column.numeric_column(m) for m in training_features.columns]
logistic_model = LinearClassifier(feature_columns=feature_columns_numeric)

logistic_model.train(train_fn_feeds)
#%%% Predicciones
testing_fn_feeds = lambda: get_input_fn(normed_train_features, training_labels)
test_fn_feeds = lambda: get_input_fn(normed_test_features, test_labels)

train_predictions = logistic_model.predict(testing_fn_feeds)
test_predictions = logistic_model.predict(test_fn_feeds)
train_predictions_series = pd.Series([p['classes'][0].decode("utf-8")   for p in train_predictions])
test_predictions_series = pd.Series([p['classes'][0].decode("utf-8")   for p in test_predictions])

train_predictions_df = pd.DataFrame(train_predictions_series, columns=['predictions'])
test_predictions_df = pd.DataFrame(test_predictions_series, columns=['predictions'])
training_labels.reset_index(drop=True, inplace=True)
train_predictions_df.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)
train_labels_with_predictions_df = pd.concat([training_labels, train_predictions_df], axis=1)
test_labels_with_predictions_df = pd.concat([test_labels, test_predictions_df], axis=1)

#%% Validación
def calculate_binary_class_scores(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred.astype('int64'))
  precision = precision_score(y_true, y_pred.astype('int64'))
  recall = recall_score(y_true, y_pred.astype('int64'))
  return accuracy, precision, recall


train_accuracy_score, train_precision_score, train_recall_score = calculate_binary_class_scores(training_labels, train_predictions_series)
test_accuracy_score, test_precision_score, test_recall_score = calculate_binary_class_scores(test_labels, test_predictions_series)

print('Training Data Accuracy (%) = ', round(train_accuracy_score*100,2))
print('Training Data Precision (%) = ', round(train_precision_score*100,2))
print('Training Data Recall (%) = ', round(train_recall_score*100,2))
print('-'*50)
print('Test Data Accuracy (%) = ', round(test_accuracy_score*100,2))
print('Test Data Precision (%) = ', round(test_precision_score*100,2))
print('Test Data Recall (%) = ', round(test_recall_score*100,2))

#%% [markdown]  
# ### Modelo logístico
#
#Segunda parte de la solución usando una red neuronal de modelo logístico con un modelo Sequential modo 2
#%%

model_seq2 = keras.models.Sequential()
## Adiciona capas una por una
model_seq2.add(keras.layers.Dense(units=16, activation='relu', input_shape=(30,)))
# Adding dropout to prevent overfitting (regularización)
model_seq2.add(keras.layers.Dropout(0.1)) # 10% out in each epoc
model_seq2.add(keras.layers.Dense(units=16, activation='relu'))
# Adding dropout to prevent overfitting (regularización)
model_seq2.add(keras.layers.Dropout(0.1))
model_seq2.add(keras.layers.Dense(units=1, activation='sigmoid'))
#%% [markdown]  
# ### Compila el modelo
#%%
model_seq2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_seq2.summary()
#plot_model(model_seq2, to_file='./img/cancer_seno.png', show_shapes=True)
#%% [markdown]
# ### Entrenamiento
# %%
history = model_seq2.fit(normed_train_features, training_labels, batch_size=32, epochs=150,validation_split = 0.2)

#%% [markdown]
# ### Predicciones
#%%
# Predicting the Test set results
y_pred = model_seq2.predict(normed_test_features)
#y_pred = (y_pred > 0.5)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <=0.5] = 0

#%% [markdown]
# ### Matriz de confusión
#%%

test_accuracy_score2, test_precision_score2, test_recall_score2 = calculate_binary_class_scores(test_labels, y_pred)

print('Test Data Accuracy (%) = ', round(test_accuracy_score2*100,2))
print('Test Data Precision (%) = ', round(test_precision_score2*100,2))
print('Test Data Recall (%) = ', round(test_recall_score2*100,2))

#%%

cm = confusion_matrix(test_labels, y_pred)
print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/test_labels.shape[0])*100))
sns.heatmap(cm,annot=True)
plt.savefig("img/heatmap.png")

#%% [markdown]
# ### Evaluación del modelo
#%%

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Entrenamiento y validación '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
#%%

plot_metric(history, 'loss')
plot_metric(history, 'accuracy')
# %%
