import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
df = pd.read_csv("data/winequality-red.csv")
target = df['quality']
inputs = df.drop(columns='quality',axis=1)
x_train,x_test,y_train,y_test= train_test_split(inputs,target,test_size=0.2,random_state=11)
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)

x=0
depth=1000

with mlflow.start_run():
    dt = DecisionTreeClassifier(ccp_alpha=x,max_depth=depth)
    dt.fit(x_train_scaled,y_train)
    y_predict = dt.predict(x_test_scaled)
    acc=accuracy_score(y_predict,y_test)
    print(acc)
    mlflow.log_param("ccp_alpha",x)
    mlflow.log_param("max_depth",depth)
    mlflow.log_metric("acc",acc)
    
    remote_server_uri = "https://dagshub.com/Sahulinkan7/mlflow_project.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store !="file":
        mlflow.sklearn.log_model(
            dt,"model",registered_model_name="DecisionTreequalityModel")
    else:
        mlflow.sklearn.log_model(dt,"dtmodel")