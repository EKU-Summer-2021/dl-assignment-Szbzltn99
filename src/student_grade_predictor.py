import tensorflow as tf
import os
import pandas as pd
from src.read_csv import csv_read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder




class SGP:
    def __init__(self):
        path = '/student-mat.csv'
        self.dataframe = pd.DataFrame(csv_read(os.getcwd() + path))
        self.inputs = self.dataframe.drop('G3', axis='columns')
        self.inputs_n = self.one_hot_encoding()
        self.target = self.dataframe['G3']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.inputs_n, self.target,
                                                                                test_size=0.33, random_state=42)

        self.model = self.__build_model()

    def le_encoding_input(self):
        inputs = self.inputs
        encoder_school = LabelEncoder()
        encoder_sex = LabelEncoder()
        encoder_address = LabelEncoder()
        encoder_famsize = LabelEncoder()
        encoder_Pstatus = LabelEncoder()
        encoder_Mjob = LabelEncoder()
        encoder_Fjob = LabelEncoder()
        encoder_reason = LabelEncoder()
        encoder_guardian = LabelEncoder()
        encoder_schoolsup = LabelEncoder()
        encoder_famsup = LabelEncoder()
        encoder_paid = LabelEncoder()
        encoder_activities = LabelEncoder()
        encoder_nursery = LabelEncoder()
        encoder_higher = LabelEncoder()
        encoder_internet = LabelEncoder()
        encoder_romantic = LabelEncoder()
        inputs['school_n'] = encoder_school.fit_transform(inputs['school'])
        inputs['sex_n'] = encoder_sex.fit_transform(inputs['sex'])
        inputs['address_n'] = encoder_address.fit_transform(inputs['address'])
        inputs['famsize_n'] = encoder_famsize.fit_transform(inputs['famsize'])
        inputs['Pstatus_n'] = encoder_Pstatus.fit_transform(inputs['Pstatus'])
        inputs['Mjob_n'] = encoder_Mjob.fit_transform(inputs['Mjob'])
        inputs['Fjob_n'] = encoder_Fjob.fit_transform(inputs['Fjob'])
        inputs['reason_n'] = encoder_reason.fit_transform(inputs['reason'])
        inputs['guardian_n'] = encoder_guardian.fit_transform(inputs['guardian'])
        inputs['schoolsup_n'] = encoder_schoolsup.fit_transform(inputs['schoolsup'])
        inputs['famsup_n'] = encoder_famsup.fit_transform(inputs['famsup'])
        inputs['paid_n'] = encoder_paid.fit_transform(inputs['paid'])
        inputs['activities_n'] = encoder_activities.fit_transform(inputs['activities'])
        inputs['nursery_n'] = encoder_nursery.fit_transform(inputs['nursery'])
        inputs['higher_n'] = encoder_higher.fit_transform(inputs['higher'])
        inputs['internet_n'] = encoder_internet.fit_transform(inputs['internet'])
        inputs['romantic_n'] = encoder_romantic.fit_transform(inputs['romantic'])
        inputs_n = inputs.drop(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                                'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                                'nursery', 'higher', 'internet', 'romantic'], axis='columns')
        return inputs_n

    def one_hot_encoding(self):
        inputs = self.le_encoding_input()
        ohe15 = OneHotEncoder(categorical_features=[15])
        ohe15.fit_transform(inputs).toarray()
        ohe16 = OneHotEncoder(categorical_features=[16])
        ohe16.fit_transform(inputs).toarray()
        ohe17 = OneHotEncoder(categorical_features=[17])
        ohe17.fit_transform(inputs).toarray()
        ohe18 = OneHotEncoder(categorical_features=[18])
        ohe18.fit_transform(inputs).toarray()
        ohe19 = OneHotEncoder(categorical_features=[19])
        ohe19.fit_transform(inputs).toarray()
        ohe20 = OneHotEncoder(categorical_features=[20])
        ohe20.fit_transform(inputs).toarray()
        ohe21 = OneHotEncoder(categorical_features=[21])
        ohe21.fit_transform(inputs).toarray()
        ohe22 = OneHotEncoder(categorical_features=[22])
        ohe22.fit_transform(inputs).toarray()
        ohe23 = OneHotEncoder(categorical_features=[23])
        ohe23.fit_transform(inputs).toarray()
        ohe24 = OneHotEncoder(categorical_features=[24])
        ohe24.fit_transform(inputs).toarray()
        ohe25 = OneHotEncoder(categorical_features=[25])
        ohe25.fit_transform(inputs).toarray()
        ohe26 = OneHotEncoder(categorical_features=[26])
        ohe26.fit_transform(inputs).toarray()
        ohe27 = OneHotEncoder(categorical_features=[27])
        ohe27.fit_transform(inputs).toarray()
        ohe28 = OneHotEncoder(categorical_features=[28])
        ohe28.fit_transform(inputs).toarray()
        ohe29 = OneHotEncoder(categorical_features=[29])
        ohe29.fit_transform(inputs).toarray()
        ohe30 = OneHotEncoder(categorical_features=[30])
        ohe30.fit_transform(inputs).toarray()
        ohe31 = OneHotEncoder(categorical_features=[31])
        ohe31.fit_transform(inputs).toarray()
        return inputs

    def __build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, input_shape=(32,), activation='relu'))
        model.add(tf.keras.layers.Dense(16,activation='tanh'))
        model.add(tf.keras.layers.Dense(21, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self):
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=20, verbose=2, validation_split=0.2, shuffle=True)
        print(self.model.evaluate(self.x_test, self.y_test))


sgp = SGP()
sgp.fit()

