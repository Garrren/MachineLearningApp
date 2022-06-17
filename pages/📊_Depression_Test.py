import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    st.set_page_config(page_title="Test It Yourself", page_icon="📈")

    st.title("Depression Test")
    st.header("Answer The Questions in the sidebar")
    st.write(
        "Your responses in the sidebar will be fed to the chosen model which will then calculate the probability of"
        " depression.")
    st.sidebar.write("Welcome to the sidebar")

    ##function to load and encode the data
    @st.cache(persist=True)
    def load():
        df = pd.read_csv('dataset_depression.csv')
        # cleaning/dropping irrelevant data
        df = df.drop('Educational Level', axis=1)
        df = df.drop('How many of the electronic gadgets', axis=1)
        df = df.drop('Which of the following best describes your term-time accommodation?', axis=1)
        # use better column names
        df.columns = df.columns.str.replace("?", "")
        df.columns = df.columns.str.replace("Gender", "Sex")
        df.columns = df.columns.str.replace("Trouble falling or staying asleep, or sleeping too much", "Troubled Sleep")
        df.columns = df.columns.str.replace('Little interest or pleasure in doing things', 'LowMotivation')
        df.columns = df.columns.str.replace("Feeling tired or having little energy", "Low Energy")
        df.columns = df.columns.str.replace("Poor appetite or overeating", "Poor Appetite/Diet")
        df.columns = df.columns.str.replace(
            "Feeling bad about yourself or that you are a failure or not have let yourself or your family down",
            "Self-Loathing")
        df.columns = df.columns.str.replace(
            "Thoughts that you would be better off dead or of hurting yourself in some way",
            "Self-Harm/Suicidal Tendencies")
        df.columns = df.columns.str.replace('Feeling down, depressed, or hopeless', 'DepressionLevel')
        df.columns = df.columns.str.replace('Do you have part-time or full-time job', 'Job')
        df.columns = df.columns.str.replace('How many hours do you spend studying each day', 'HoursStudying')
        df.columns = df.columns.str.replace('How many hours do you spend on social media per day', 'TimeSpentSM')
        df.columns = df.columns.str.replace('Your Last Semester GPA:', 'GPA')
        df.loc[df['DepressionLevel'] <= 2, 'DepressionLevel'] = 0
        df.loc[df['DepressionLevel'] >= 3, 'DepressionLevel'] = 1
        label = LabelEncoder()

        df['Age'] = label.fit_transform(df['Age'])
        df['Sex'] = label.fit_transform(df['Sex'])
        df['HoursStudying'] = label.fit_transform(df['HoursStudying'])
        df['Job'] = label.fit_transform(df['Job'])
        df['TimeSpentSM'] = label.fit_transform(df['TimeSpentSM'])

        return df

    df = load()

    # creating the classifier
    def get_classifier(clf_name):
        if clf_name == "SVM":
            clf = SVC(C=5.99, probability=True)
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100,
                                         max_depth=100, random_state=1234)
        else:
            clf = KNeighborsClassifier(n_neighbors=1)
        return clf

    classifier_name = st.sidebar.selectbox("Select which Classifier to use", ("SVM", "KNN",
                                                                              "Random Forest"))

    clf = get_classifier(classifier_name)

    # splitting X and y
    X = df.drop('DepressionLevel', axis=1)
    y = df['DepressionLevel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)

    clf.fit(X_train, y_train)

    # accepting user input by utilizing streamlits sliders and selectors
    def user_input_features():
        gender = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        age = st.sidebar.selectbox('Age', ('19 to 24 years', '18 years or less', '25 years and above'))
        mot = st.sidebar.slider('Low Motivation', 1, 4, 2)
        slp = st.sidebar.slider('Troubled Sleep', 1, 4, 2)
        ene = st.sidebar.slider('Low Energy', 1, 4, 2)
        eat = st.sidebar.slider('Poor Appetite', 1, 4, 2)
        sl = st.sidebar.slider('Self-Loathing', 1, 4, 2)
        conc = st.sidebar.slider('Trouble Concentrating', 1, 4, 2)
        mov = st.sidebar.slider('Moving or speaking so slowly that other people could have noticed '
                                'Or being so restless that you have been moving around a lot more than usual', 1, 4, 2)
        job = st.sidebar.selectbox('Job', ("No", "Part Time", "Full Time"))
        sh = st.sidebar.slider('Self-Harm/Suicidal Tendencies', 1, 4, 2)
        std = st.sidebar.selectbox('Time Spent Studying', ("1-2 Hours", "2-4 Hours", "More than 4 hours"))
        soc = st.sidebar.selectbox('Time Spent on Social Media', ("1-2 Hours", "2-4 Hours", "More than 4 hours"))
        gpa = st.sidebar.slider("Gpa", 1.0, 5.0, 2.0)

        # assigning the user input to data
        data = {'Sex': gender,
                'Age': age,
                'LowMotivation': mot,
                'Troubled Sleep': slp,
                'Low Energy': ene,
                'Poor Appetite/Diet': eat,
                'Self-Loathing': sl,
                'Trouble concentrating on things, such as reading the newspaper or watching television': conc,
                'Moving or speaking so slowly that other people could have noticed Or being so restless that you have been moving around a lot more than usual': mov,
                'Self-Harm/Suicidal Tendencies': sh,
                'Job': job,
                'HoursStudying': std,
                'TimeSpentSM': soc,
                'GPA ': gpa}

        # transforming data into a dataframe
        features = pd.DataFrame(data, index=[0])
        return features

    test = user_input_features()

    # encode qualitative data
    label = LabelEncoder()

    test['Age'] = label.fit_transform(test['Age'])
    test['Sex'] = label.fit_transform(test['Sex'])
    test['HoursStudying'] = label.fit_transform(test['HoursStudying'])
    test['Job'] = label.fit_transform(test['Job'])
    test['TimeSpentSM'] = label.fit_transform(test['TimeSpentSM'])

    # make prediction and print probability of depression
    prediction = clf.predict(test)
    prediction_proba = clf.predict_proba(test)

    st.subheader('Prediction')
    murung = np.array(['Not Depressed', 'Depressed'])
    st.write(murung[prediction])

    st.subheader('Prediction Probability')
    st.write('0 = Not Depressed')
    st.write('1 = Depressed')
    st.write(prediction_proba)
    st.write('If your probability of being depressed is 0.8 or above, it is highly recommended you seeek profesional help.')

    st.write(
        "Mental Health Should Not Be Neglected. If you ever feel the need to ask for help. Contact the hotline below.")
    st.write('MALAYSIAN MENTAL HEALTH ASSOCIATION (MMHA)')
    st.write('Contact Number:03-2780 6803')
    st.write('E-Mail: admin@mmha.org.my')
    st.write('Website: https://mmha.org.my/')


if __name__ == '__main__':
    main()
