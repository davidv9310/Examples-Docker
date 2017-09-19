
# coding: utf-8

# In[2]:

from daal.data_management import FileDataSource
from daal.data_management import HomogenNumericTable
from daal.data_management import DataSourceIface
from daal.data_management import NumericTableIface

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing

dataset_filename = 'iris.data'
iris_datasource = FileDataSource(dataset_filename, DataSourceIface.doAllocateNumericTable, 
                                 DataSourceIface.doDictionaryFromContext)
number_of_observations = iris_datasource.loadDataBlock()

print("Observations read: {}".format(number_of_observations))


# In[3]:

data_to_pandas = pd.read_csv('iris.data', delimiter=',', names=['sepal_length', 'sepal_width',
                                                                'petal_length', 'petal_width', 
                                                                'class'])

data_array = data_to_pandas.values
numpy_array = data_to_pandas.transpose()[0: 4].transpose().values
numpy_array_targets = data_to_pandas.transpose()[4: 5].transpose().values
print("Is C Contiguous? {}".format(numpy_array.flags['C']))

# Important! To crate a HomogenNumericTable the array must be contiguous.
numpy_array = np.ascontiguousarray(numpy_array, dtype=np.double)
print("Is C Contiguous? {}".format(numpy_array.flags['C']))

array_nt = HomogenNumericTable(numpy_array)
print("Observations read: {}".format(array_nt.getNumberOfRows()))


# In[4]:

iris_targets = data_to_pandas.transpose()[4: 5].transpose()

le_data = preprocessing.LabelEncoder()
le_data.fit(iris_targets.values.ravel())
iris_target_encoded = le_data.transform(iris_targets.values.ravel())
iris_target_encoded.shape = (150, 1)

data_array[:, :5] = iris_target_encoded
data_array[:, :-1] = numpy_array

print("Is C Contiguous? {}".format(data_array.flags['C']))

data_array = np.ascontiguousarray(data_array, dtype=np.double)
print("Is C Contiguous? {}".format(data_array.flags['C']))

array_nt_data = HomogenNumericTable(data_array)
print("Observations read: {}".format(array_nt_data.getNumberOfRows()))


# In[5]:

print("Dimensions: ({},{})".format(array_nt.getNumberOfRows(), array_nt.getNumberOfColumns()))
print("Dimensions: ({},{})".format(array_nt_data.getNumberOfRows(), array_nt_data.getNumberOfColumns()))


# In[6]:

from daal.data_management import BlockDescriptor_Float64, readOnly, HomogenNumericTable

block_descriptor = BlockDescriptor_Float64()
array_nt.getBlockOfRows(0, array_nt.getNumberOfRows(), readOnly, block_descriptor)
numpy_array2 = block_descriptor.getArray()

array_nt.releaseBlockOfRows(block_descriptor)
print("Dimensions: {}".format(numpy_array2.shape))

block_descriptor_data = BlockDescriptor_Float64()
array_nt_data.getBlockOfRows(0, array_nt_data.getNumberOfRows(), readOnly, block_descriptor_data)
data_array2 = block_descriptor_data.getArray()

array_nt_data.releaseBlockOfRows(block_descriptor_data)
print("Dimensions: {}".format(data_array2.shape))


# In[7]:

average_setosa = []
average_versicolor = []
average_virginica = []

for row in data_array:
    if row[4] == 0.0:
        average_setosa.append(row)
    elif row[4] == 1.0:
        average_versicolor.append(row)
    elif row[4] == 2.0:
        average_virginica.append(row)
    else: 
        continue


datos_setosa = pd.DataFrame(average_setosa)
datos_versicolor = pd.DataFrame(average_versicolor)
datos_virginica = pd.DataFrame(average_virginica)

mean_setosa = datos_setosa.mean()
mean_versicolor = datos_versicolor.mean()
mean_virginica = datos_virginica.mean()
print(mean_setosa)
# print(mean_versicolor)
# print(mean_virginica)


# In[8]:

sample = np.random.choice(len(numpy_array), size=math.floor(.8*len(numpy_array)), replace=False)
select = np.in1d(range(numpy_array.shape[0]), sample)

numpy_train_data = numpy_array[select,:]
numpy_test_data = numpy_array[~select,:]

train_data_table = HomogenNumericTable(numpy_train_data)
test_data_table = HomogenNumericTable(numpy_test_data)

print("Number of Observations in Training Partition: {}".format(train_data_table.getNumberOfRows()))
print("Number of Observations in Test Partition: {}".format(test_data_table.getNumberOfRows()))


# In[9]:

sample_data = np.random.choice(len(data_array), size=math.floor(.8*len(data_array)), replace=False)
select_data = np.in1d(range(data_array.shape[0]), sample_data)

numpy_train_data2 = data_array[select,:]
numpy_test_data2 = data_array[~select,:]

train_data_table2 = HomogenNumericTable(numpy_train_data2)
test_data_table2 = HomogenNumericTable(numpy_test_data2)

print("Number of Observations in Training Partition: {}".format(train_data_table2.getNumberOfRows()))
print("Number of Observations in Test Partition: {}".format(test_data_table2.getNumberOfRows()))


# In[10]:

train_data_dependent_variable = numpy_train_data2[:, 4].reshape(120,1)
print("Dimension of the train_data dependent variable: {}".format(train_data_dependent_variable.shape))

train_data = np.delete(numpy_train_data2, (0, 1, 4), axis=1)
print("Dimension of the train_data: {}".format(train_data.shape))

# We need to do the same with the test_data

test_data_dependent_variable = numpy_test_data2[:, 4].reshape(30,1)
print("Dimension of the test_data dependent variable: {}".format(test_data_dependent_variable.shape))

test_data = np.delete(numpy_test_data2, (0, 1, 4), axis=1)
print("Dimension of the test_data: {}".format(test_data.shape))


# In[11]:

from daal.algorithms.linear_regression import training

train_data = np.ascontiguousarray(train_data, dtype=np.double)
print("Is C Contiguous? {}".format(train_data.flags['C']))
train_data_table = HomogenNumericTable(train_data)

train_data_dependent_variable = np.ascontiguousarray(train_data_dependent_variable, dtype=np.double)
print("Is C Contiguous? {}".format(train_data_dependent_variable.flags['C']))
train_outcome_table = HomogenNumericTable(train_data_dependent_variable)

algorithm = training.Batch_Float64NormEqDense()
algorithm.input.set(training.data, train_data_table)
algorithm.input.set(training.dependentVariables, train_outcome_table)
trainingResult = algorithm.compute()

beta = trainingResult.get(training.model).getBeta()
block_descriptor = BlockDescriptor_Float64()
beta.getBlockOfRows(0, beta.getNumberOfRows(), readOnly, block_descriptor)
beta_coeficients = block_descriptor.getArray()
beta.releaseBlockOfRows(block_descriptor)

print("Coeficients: {}".format(beta_coeficients))


# In[12]:

np.savetxt('Coeficients_Model.csv', beta_coeficients, delimiter=',')
print('Success')


# In[13]:

from daal.algorithms.linear_regression import prediction

# print(test_data)
test_data = np.ascontiguousarray(test_data, dtype=np.double)
print("Is C Contiguous? {}".format(test_data.flags['C']))
test_data_table = HomogenNumericTable(test_data)

# print(test_data_dependent_variable)
test_data_dependent_variable = np.ascontiguousarray(test_data_dependent_variable, dtype=np.double)
print("Is C Contiguous? {}".format(test_data_dependent_variable.flags['C']))
test_outcome_table = HomogenNumericTable(test_data_dependent_variable)

algorithm = prediction.Batch()
algorithm.input.setTable(prediction.data, test_data_table)
algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

predictionResult = algorithm.compute()

test_result_data_table = predictionResult.get(prediction.prediction)

block_descriptor = BlockDescriptor_Float64()

test_result_data_table.getBlockOfRows(0, test_result_data_table.getNumberOfRows(), readOnly, block_descriptor)

numpy_test_result = block_descriptor.getArray()

test_result_data_table.releaseBlockOfRows(block_descriptor)

prediction = numpy_test_result.round()

print("First 4 predicted values:")
print(prediction[1:30])
print("First 4 real responses:")
print(test_data_dependent_variable[1:30])


# In[14]:

trainingResult.get(training.model)


# In[15]:

def RMSE(y, y_prime):
    return np.sqrt(np.mean((y_prime - y)**2))

print("RMSE: {}".format(RMSE(test_data_dependent_variable, numpy_test_result)))


# In[16]:

from daal.algorithms.ridge_regression import training
from daal.algorithms.ridge_regression import prediction
from sklearn.preprocessing import PolynomialFeatures

train_outcome_table = HomogenNumericTable(train_data_dependent_variable)

for i in range(1,5):
    poly = PolynomialFeatures(degree=i)

    expanded_train_data = poly.fit_transform(train_data)
    train_data_table = HomogenNumericTable(expanded_train_data)    

    algorithm = training.Batch()
    algorithm.input.set(training.data, train_data_table)
    algorithm.input.set(training.dependentVariables, train_outcome_table)

    training_model = algorithm.compute().get(training.model)

    expanded_test_data = poly.fit_transform(test_data)

    test_data_table = HomogenNumericTable(expanded_test_data)
    test_outcome_table = HomogenNumericTable(test_data_dependent_variable)

    prediction_algorithm = prediction.Batch()

    prediction_algorithm.input.setNumericTableInput(prediction.data, test_data_table)
    prediction_algorithm.input.setModelInput(prediction.model, training_model)

    prediction_result = prediction_algorithm.compute()

    test_result_data_table = prediction_result.get(prediction.prediction)
    block_descriptor = BlockDescriptor_Float64()
    test_result_data_table.getBlockOfRows(0, test_result_data_table.getNumberOfRows(), 
                                              readOnly, block_descriptor)
    numpy_test_result = block_descriptor.getArray()
    test_result_data_table.releaseBlockOfRows(block_descriptor)

    print("{} degree of expanstion RMSE: {}".format(i, RMSE(test_data_dependent_variable,numpy_test_result)))


# In[17]:

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

for i in range(5,6):
    poly = PolynomialFeatures(degree=i)
    expanded_train_data = poly.fit_transform(train_data)

    regr = linear_model.LinearRegression()

    regr.fit(expanded_train_data, train_data_dependent_variable)

    numpy_test_result = regr.predict(poly.fit_transform(test_data))
    
    print("{} degree of expanstion RMSE: {}".format(i, RMSE(test_data_dependent_variable,numpy_test_result)))


# In[ ]:



