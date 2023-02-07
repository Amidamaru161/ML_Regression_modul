from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import joblib
from joblib import dump, load
from scipy.stats import zscore
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def read_file():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    if filename.endswith('.csv'):
        df=pd.read_csv(filename,encoding='unicode_escape',sep=";")
        #fill NaNs with column means in each column 

        return  df
    elif filename.endswith('.xlsx'):
        df=pd.read_excel(filename)
        return df
    else:
        raise ValueError('File type not supported. Please select a CSV or XLSX file.')
def read_dumpfile():
    Tk().withdraw()
    file_dump = askopenfilename()
    if file_dump.endswith('.joblib'):
        loaded_model = load(file_dump)
        return loaded_model
    else:
        raise ValueError('File type not supported.')
print('Enter filename of model')
loaded_model=read_dumpfile()

print('Enter filename of excel with unpredicted data: ')
data = read_file()
def nune(data):
    for col in data.columns:
        data[col]=pd.to_numeric(data[col], errors='coerce')
    data=data.dropna()
    return data


data=nune(data)


question_for_target=input('You hav target name?(Y/N)')
    
if question_for_target=='Y':

    target_name=input('Enter target name: ')
    target_name_df=data[target_name]
    data=data.drop(target_name,axis=1)



    data['Predicted_data'] = loaded_model.predict(data)
    data['Real_data'] =target_name_df    

    mse = mean_squared_error(data['Real_data'], data['Predicted_data'])
    r2=metrics.r2_score(data['Real_data'], data['Predicted_data'])
    rmse= np.sqrt(mse)

    my_file = open("metric_model.txt", "w+")
    my_file.write(f"MODEL={loaded_model[1][1]}\n")
    my_file.write(f"MSE={mse}\n")
    my_file.write(f"RMSE={rmse}\n")
    my_file.close()

    output_name_unpredicted = input('Enter output filename of excel (EXAMPLE.XLSX): ')
    data.to_excel(f"{output_name_unpredicted}.xlsx")
else :
    output_name_unpredicted = input('Enter output filename of excel (EXAMPLE.XLSX): ')
    data['Predicted_data'] = loaded_model.predict(data)
    
    data.to_excel(f"{output_name_unpredicted}.xlsx")
    my_file = open("metric_model.txt", "w+")
    my_file.write(f"MODEL={loaded_model[1][1]}\n")
    my_file.close()