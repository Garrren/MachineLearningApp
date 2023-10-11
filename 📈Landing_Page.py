import subprocess
subprocess.call(['conda', 'install', 'matplotlib'])
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    # writing to the streamlit page
    st.title("Depression Indicator")

    st.header("Welcome Page")
    st.write(
        "Depression is more than a common occurence nowadays, we've prepared this app to analyse your risk of suffering from depression"
        " and to spread the awareness of depression especially among students.")
    st.write("We must however first understand how we are ale to predict the likelihood of depression.")
    st.write("To start. Let's navigate to the sidebar.")
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

        # encoding qualitative data

        label = LabelEncoder()
        df['Age'] = label.fit_transform(df['Age'])
        df['Sex'] = label.fit_transform(df['Sex'])
        df['HoursStudying'] = label.fit_transform(df['HoursStudying'])
        df['Job'] = label.fit_transform(df['Job'])
        df['TimeSpentSM'] = label.fit_transform(df['TimeSpentSM'])

        return df

    df = load()

    # getting parameters
    def getparam(clf_name):
        st.sidebar.caption("Play around with the parameters to get the best accuracy.")
        params = dict()
        if clf_name == "SVM":
            c = st.sidebar.slider("C", 0.1, 10.0)
            params["C"] = c
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("No_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        else:
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        return params

    # creating the classifier
    def get_classifier(clf_name, params):
        if clf_name == "SVM":
            clf = SVC(C=params["C"])
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                         max_depth=params["max_depth"], random_state=1234)
        else:
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        return clf

    st.sidebar.write("To see the dataset we used for this classifier, click on the checkbox.")

    if st.sidebar.checkbox("Display data", "False"):
        st.subheader("Depression dataset")
        st.write(df)

    classifier_name = st.sidebar.selectbox("Select which Classifier to use", ("SVM", "KNN",
                                                                              "Random Forest"))
    params = getparam(classifier_name)

    clf = get_classifier(classifier_name, params)

    # splitting X and y
    X = df.drop('DepressionLevel', axis=1)
    y = df['DepressionLevel']   #target

    # splitting into test and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)

    # fitting the model, making the prediction and getting the accuracy score
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    # printing the accuracy of the classifier
    st.write(f"Classifier = {classifier_name}")
    st.write(f"Accuracy = {acc} %")

    st.write("Now that you have a better understanding of the algorithm, let's proceed to the test.")

    class_names = ['Not Depressed', 'Depressed']

    # plotting the metrics
    metrics = st.sidebar.multiselect("What metrics to plot?",
                                     ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    if 'Confusion Matrix' in metrics:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(clf, X_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics:
        st.subheader("ROC Curve")
        plot_roc_curve(clf, X_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(clf, X_test, y_test)
        st.pyplot()


if __name__ == '__main__':
    main()
