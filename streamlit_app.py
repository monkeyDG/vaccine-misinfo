"""
Created by David Gallo (https://github.com/monkeydg/)

Vaccine Misinformation Checker

This code is the front-end web interface used to interact with the machine learning server.
It is deployed on an AWS EC2 t2.micro instance.
"""

import urllib
import requests
import pandas as pd
import streamlit as st

FAVICON_PATH = "./assets/favicon.png"
HEADER_IMAGE_PATH = "./assets/header_image.jpg"
SCORES_PATH = "./pkls/scores.pkl"
PARAMS_PATH = "./pkls/params.pkl"

def activate_css():
    """Define new CSS classes for custom text highlighting in Streamlit."""
    # we could put this into a styles.css file and import it,
    # but with only a handful of css classes this is simpler.
    st.markdown("""
    <style>
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    a:link , a:visited{
        color: dimgray;
    }
    a:hover,  a:active {
        color: black;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: dimgray;
        text-align: center;
    }
    .h1 {
        font-size:80px !important;
    }
    .h2 {
        font-size:20px !important;
    }
    .bold {
        font-weight: bold !important;
    }
    .green {
        color: #006e0f !important;
    }
    .red {
        color: #8f1800 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def color_results(val):
    """Helper function to color the output of show_scores"""
    color = 'green' if val >= 0.95 else 'black'
    return f'color: {color}'  # css color formatting

def show_scores(df_scores):
    """Pretty-prints a table with the scores of each classifier in a list of Classifier objects."""
    df_scores.drop(columns=['shortname'], inplace=True)  # we don't care about the short clf name
    rounded = dict.fromkeys(df_scores.select_dtypes('float').columns, "{:.4}")
    highlight = {"AdaBoost": 'background-color: #c6ffba; font-weight: bold;'}
    df_scores = df_scores.style.\
        applymap(color_results, subset=['F1']).\
        format(rounded).\
        apply(lambda x: x.index.map(highlight), subset=['F1'])
    st.dataframe(df_scores, height=1000)

def display_dashboard(scores):
    """Displays the streamlit dashboard page where users can enter a search string or tweet"""
    st.subheader('Check for vaccine misinformation')
    input = st.text_input(
        """Insert your text here, or a link to a tweet
        (e.g. https://twitter.com/user/status/1234567890)"""
        )
    
    with st.expander("Change the classification algorithm..."):
        st.write("""
        You can see how different machine learning models compare by checking out the
        'Machine learning classifiers' page using the navigation menu.
        """)

        # todo: disable k-means in grey and delete the .pkl file so its never loaded

        selected_clf = st.radio(
            "Choose a classifier (default = AdaBoost)",
            scores.index,
            index=8  # index of AdaBoost in the list
            )
        
        if selected_clf == "K-Nearest Neighbors":
            st.error(
                """K-nearest Neighbors is currently unavailable as it is too memory-intensive.
                AdaBoost will be used instead.
                """)
            selected_clf = "AdaBoost"
        
        # gets the shortened name of the selected classifier
        # since this is the filename of the pkl file we need to load
        selected_clf = scores.loc[selected_clf, 'shortname']
    
    run_analysis = st.button('Check text')
    
    selected_clf = "s_clf"  # set default classifier
    if run_analysis:
        with st.spinner("Evaluating..."):
            try:
                get_misinformation(input, selected_clf)
            except Exception as e:
                st.error("Error connecting to the ML server backend.")
                with st.expander("Click here to see the exception that was raised..."):
                    st.write(e)

    st.markdown('#')  # some whitespace after the button

def get_misinformation(input, selected_clf):
    """Queries the Flask server for misinformation predictions and displays the result"""
    if input.startswith("http"):  # if we are passed a twitter url
        response = requests.get(f"http://127.0.0.1:5000/predict?clf={selected_clf}&url={input}")
    else:  # else we use raw text
        response = requests.get(f"http://127.0.0.1:5000/predict?clf={selected_clf}&text={input}")
    if response.status_code == 200:
        result = response.json()['is_misinformation']
        if result == 0:
            st.markdown("<p class='h1 green'>SAFE</p>", unsafe_allow_html=True)
            st.success("The machine learning model did not detect any potential vaccine misinformation.")
        elif result == 1:
            st.markdown("<p class='h1 red'>UNSAFE</p>", unsafe_allow_html=True)
            st.error("The machine learning model detected potential vaccine misinformation.")
        elif result == "Bad URL":
            st.markdown(
                """Unable to parse the tweet at {input}.
                Please ensure your tweet url is formatted correctly."""
                )
        elif result == "NA":
            st.warning("Please enter text or a tweet to evaluate...")
        else:
            st.markdown("Error: unexpected response from ML server.")
    else:
        st.write(f'Error querying ML server, HTTP status code {response.status_code}')

def display_source_code():
    """Streamlit layout for the source code page with the interactive_stats.py displayed."""
    st.subheader('Source code')
    st.write(
        """<p>
        Full source code and training data can be found at:
        <a href="https://github.com/monkeydg/vaccine-misinformation/" target="_blank">
        https://github.com/monkeydg/vaccine-misinformation/</a>
        </p>""",
        unsafe_allow_html=True
        )
    # query from github to allow us to prevent accidentally exposing something
    ml_url = "https://raw.githubusercontent.com/monkeydg/POG-bot/20c6aba563d1ef3955d9ab0b269c70f34d7eda3a/bot/modules/interactive_stats.py"
    with urllib.request.urlopen(ml_url) as response:
        ml_text = response.read().decode("utf-8")
    
    with st.expander("Machine learning Flask server handling requests on the back-end"):
        st.code(ml_text)

    st_url = "https://raw.githubusercontent.com/monkeydg/POG-bot/20c6aba563d1ef3955d9ab0b269c70f34d7eda3a/bot/modules/interactive_stats.py"
    with urllib.request.urlopen(st_url) as response:
        st_text = response.read().decode("utf-8")
    
    with st.expander("Streamlit interface on the front-end"):
        st.code(st_text)

def display_rest_api(scores):
    """Streamlit layout for the REST API details page."""
    st.subheader('REST API')
    st.write("""
    Want to build your own UI? You can query the machine learning model directly with a GET request
    via my REST API, which is exposed at http://tbs-aiba.ml:5000/predict
    \nThe request body should be a JSON object with a 'text' or 'url' key, and optionally
    the shortname of a classifier with the 'clf' key. The response will be a JSON object with a
    'is_misinformation' key or error message. The 'is_misinformation' key will be 0 if the text is
    not misinformation, 1 if it is.
    \nHere's an example of how to format a request with a link to a tweet in Python:
    """)
    st.code(
        """import requests\nresponse = requests.get('http://127.0.0.1:5000/predict?clf=s_clf&url=https://twitter.com/user/status/1234567890')"""
        )
    st.write("And here's a request with just some text to check:")
    st.code(
        "response = requests.get('http://127.0.0.1:5000/predict?clf=s_clf&text=This is some text to check')"
        )
    st.write(
        """To select a specific classifier, pass the shortname based on the table below. 
        If none is selected, AdaBoost will be used."""
    )
    st.table(scores[['shortname']])

def display_ml_classifiers(scores, params):
    """Streamlit layout for the machine learning classifiers details and comparison page."""
    st.write("""
    Here are the scoring metrics for each classifier.
    I used F1 score as my metric of choice when selecting a best model.
    Any F1 score >0.95 is considered a excellent.
    """)
    show_scores(scores)

    st.write("""
    Here are the parameters I used for each classifier:
    """)
    st.table(params)

    st.write("""
    The hyperparameters for the top-performing models tuned using a
    gridsearch algorithm searching the following grid space:
    """)

    st.code("""
param_decisiontree = { 
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 1, 2, 4, 5, 10, 15, 20],
    'min_samples_split': [2, 4, 6, 8, 10, 20, 30, 40],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30] 
}
param_logistic = { 
    'C': np.logspace(-4, 4, 50),
    'penalty': ['l1', 'l2'], 
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'class_weight': ['balanced', None]
}
param_gaussian = {
    'var_smoothing': np.logspace(0, -9, num = 100)
}
param_random = {
    'min_samples_split': [3,5,10],
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 15],
    'max_features': [3, 5, 9],
    'bootstrap': [True, False]
}
param_svc = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1]
}
param_ada = {
    'n_estimators' : [10, 50, 100, 500],
    'learning_rate' : [0.0001, 0.001, 0.01, 0.1, 1.0]
}
param_sgd = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"] 
}
param_kn = {
    'n_neighbors':[3,5,11,19],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']
    
}
param_gboost = {'learning_rate': [0.01,0.02,0.04],
                'subsample'    : [0.9, 0.5, 0.2],
                'n_estimators' : [100,500, 1500],
                'max_depth'    : [4,8,10]
}
    """)

def main():
    st.set_page_config(
        page_title="Vaccine Misinformation Checker",
        page_icon=FAVICON_PATH
    )
    activate_css()

    st.image(HEADER_IMAGE_PATH)
    st.title('Vaccine Misinformation Checker')

    st.sidebar.title("Page Selector")
    app_mode = st.sidebar.selectbox(
        "For developers",
        ["Dashboard", "Machine learning classifiers", "REST API", "Source code"]
        )

    st.sidebar.title("About")
    st.sidebar.info(
        """
        Built and deployed by [David Gallo](https://www.linkedin.com/in/davidgallo747/).
        \nThis AI app was trained using 6000 labeled vaccine misinformation tweets, then uses
        NLTK NLP methods and Scikit-learn's classification algorithms to analyze new tweets/text.
        """)

    st.sidebar.title("Contribute")
    st.sidebar.info(
        """
        This an open source project, so you're welcome to contribute your own comments,
        issues, or features by creating a [pull requests on github](https://github.com/monkeydg/vaccine-misinformation/).
        """)

    scores = pd.read_pickle(SCORES_PATH)
    params = pd.read_pickle(PARAMS_PATH)

    if app_mode == "Dashboard":
        display_dashboard(scores)
    elif app_mode == "Machine learning classifiers":
        display_ml_classifiers(scores, params)
    elif app_mode == "REST API":
        display_rest_api(scores)
    elif app_mode == "Source code":
        display_source_code()

if __name__ == '__main__':
    main()
