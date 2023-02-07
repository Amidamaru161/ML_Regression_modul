from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
from scipy.stats import zscore
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import FunctionTransformer





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

df=read_file()
#sheet_name_input=input('Enter sheet_name in excel: ')
target_name=input('Enter target name: ')


def nune(data):
    for col in data.columns:
        data[col]=pd.to_numeric(data[col], errors='coerce')
    data=data.dropna()
    return data


df=nune(df)
y=df[target_name]
X=df.drop(target_name,axis=1)




numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()   
categorical_columns = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_column_indices = [X.columns.get_loc(col) for col in numerical_columns]
cat_column_indices = [X.columns.get_loc(col) for col in categorical_columns]

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot',OneHotEncoder(handle_unknown='ignore', sparse=False))
])

num_pipeline = Pipeline(steps=[
    
    ('impute', SimpleImputer(strategy='mean')),
    ('one-hot',StandardScaler())
])

col_trans = ColumnTransformer(transformers=[
   
    ('num_pipeline',num_pipeline,num_column_indices),
    ('cat_pipeline',cat_pipeline,cat_column_indices)
    ],
    remainder='drop',
    n_jobs=-1)

""" 
model1=SVR()
param1={}
param1['classifier__kernel']=['linear', 'rbf']
param1['classifier__C']=np.logspace(-1,4,180)
param1['classifier'] = [model1] """

model2=MLPRegressor()
param2={}
param2['classifier__activation']=['identity', 'logistic', 'tanh', 'relu']
param2['classifier__solver']=['lbfgs', 'sgd', 'adam']
param2['classifier__max_iter']=[100,200,500]
param2['classifier__learning_rate_init']=[0.01,0.1,10,1]
param2['classifier']=[model2]


pipeline = Pipeline(steps=[('preprocessor', col_trans), ('classifier', model2)])
params = [param2]

gs = GridSearchCV(pipeline, params, n_jobs=-1,verbose=5, scoring='neg_mean_squared_error').fit(X, y)
clf=gs.best_estimator_
best_params=gs.best_params_



clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    ('model', clf)
])


model_pipline=clf_pipeline.fit(X, y)
predict=model_pipline.predict(X)

df['Predict_data'] = predict
df['Real_data']=y

mse = mean_squared_error(y, predict)
r2=metrics.r2_score(y,predict)
rmse= np.sqrt(mse)


savemodelquestion = input('Save model?(Y/N):')

if savemodelquestion== 'Y':
    file_name_save_model = input('Enter filename of model(example.joblib): ')
    dump(model_pipline, f"{file_name_save_model}.joblib")
    #output_name_unpredicted = input('Enter output filename of excel (EXAMPLE.XLSX): ')
    df.to_excel('predict.xlsx')
    my_file = open("param_model.txt", "w+")
    my_file.write(f"BEST MODEL={best_params}\n")
    my_file.write(f"MSE={mse}\n")
    my_file.write(f"RMSE={rmse}\n")
    my_file.close()

exitquestion = input('enter Y when exit:')
if exitquestion == 'Y':
    exit()
