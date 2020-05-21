import numpy as np
import csv 
from UtilityFunctions import ReadData_char
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


train_features, train_labels, test_features = ReadData_char()

ordinalEncoder = OrdinalEncoder(dtype= np.int64)
ordinalEncoder.fit(train_features)
train_enc = ordinalEncoder.transform(train_features)
test_enc = ordinalEncoder.transform(test_features)

one = OneHotEncoder()
one.fit(train_enc)
train_enc = one.transform(train_enc).toarray()
test_enc = one.transform(test_enc).toarray()

X_train, X_test, y_train, y_test = train_test_split(train_enc, train_labels, test_size= 0.2)

def neuralNet():
    
    std_scaler = StandardScaler()
    
    std_scaler.fit(X_train)

    X_train_trsf = std_scaler.transform(X_train)
    X_test_trsf = std_scaler.transform(X_test)

    mlpclass = MLPClassifier(hidden_layer_sizes=(1000), verbose=10, max_iter=120, tol=0.00001, alpha=0.000001, batch_size='auto', random_state=10)

    mlpclass.fit(X_train_trsf,y_train)

    y_pred = mlpclass.predict(X_test_trsf)
    
    return y_pred
        
y_pred = neuralNet()

print('F1-Score : ',f1_score(y_test,y_pred))

output_data = []

for i in range (len(y_pred)):
    output_data.append([y_pred[i]])
    
#we return the data to the csv file   
filename = "result_test_SGS.csv"
  
# writing to csv file 
with open(filename, 'w') as csvfile:
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)    
    # writing the data rows 
    csvwriter.writerows(output_data)
