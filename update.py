import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#read the data from csv file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')

#seprate the columns for making dummy variables
gender = dataset['gender']
ssc_b = dataset['ssc_b']
hsc_b = dataset['hsc_b']
hsc_s = dataset['hsc_s']
degree_t = dataset['degree_t']
workex = dataset['workex']
specialisation = dataset['specialisation']
status = dataset['status']

#creating dummy variables
gender = pd.get_dummies(gender, drop_first=True )
ssc_b = pd.get_dummies(ssc_b, drop_first=True )
hsc_b = pd.get_dummies(hsc_b, drop_first=True )
hsc_s = pd.get_dummies(hsc_s, drop_first=True)
degree_t = pd.get_dummies(degree_t, drop_first=True)
workex = pd.get_dummies(workex, drop_first=True )
specialisation = pd.get_dummies(specialisation, drop_first=True)
status = pd.get_dummies(status)

#assign the values to variables X and y
X = dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
X = pd.concat([X, gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation], axis=1)
y = status['Placed']

#separate the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#create the model
model = Sequential()
model.add(Dense(units=20, input_dim=14, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units=1, activation='sigmoid'))


#compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

#model fitting
epoch = 400
model.fit(X_train, y_train,epochs= epoch, verbose=0)

#Evaluate accuracy of the model
result = model.evaluate(X_test, y_test, verbose=1)
print("test accuracy: ", result[1])
file = open("accuracy","w+")
file.write(result[1])
file.close
