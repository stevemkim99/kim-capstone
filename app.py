import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title = "Breast Cancer Dataset Explorer", page_icon = ":hospital:")

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "EDA", "Modeling", "Make Predictions"])

df = pd.read_csv('data/breast-cancer.csv')

# HOMEPAGE
if page == "Home":
    st.title(":hospital: Breast Cancer Dataset Explorer App")
    st.subheader("Welcome to our Breast Cancer Dataset explorer app!")
    st.write("This app is designed to make the exploration and analysis of the breast cancer dataset easy and accessible to all!")
    st.image("https://www.statnews.com/wp-content/uploads/2022/03/AdobeStock_246942922.jpeg")
    st.write("Use the sidebar to navigate between different sections!")

# DATA
    
if page == "Data Overview":
    st.title(":1234: Data Overview")
    st.subheader("About the Data")
    st.write("This is one of the datasets that grabbed reports on patients breast cancer/information on their tumor.")
    st.image("https://images.squarespace-cdn.com/content/v1/5c7d3465c2ff614ea868a3c4/1601524296995-T6YC2VCSVUCNSLUYNGM6/woman-hands-holding-pink-breast-cancer-awareness-ribbon-white_53476-3876+%281%29.jpg")
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data", help = "Breast Cancer Dataset Kaggle Page")

    st.subheader("Quick Glance at the Data")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.write("In diagnosis, M = Malignant and B = Benign")
        st.dataframe(df)


    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")

        if st.toggle('Further breakdown of columns'):
            num_cols = df.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns{obj_cols}")

    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}")
        st.write(f"There are {df.shape[0]} rows (Patients) and {df.shape[1]} columns (Tumor Information).")

# EDA 

if page == "EDA":
    st.title(":bar_chart: EDA")
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df.select_dtypes(include = 'object').columns.tolist()

    eda_type = st.multiselect("What type of EDA are you interested in exploring?",
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Barplot'])
    
    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your histogram:", num_cols, index = None)

        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Diagnosis Hue on Histogram"):
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title, color = 'diagnosis', barmode = 'overlay'))
            else: 
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title))


    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("Boxplots Visualizing Numerical Distribtutions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
       
        if b_selected_col:           
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Diagnosis Hue on Box Plot"):
                st.plotly_chart(px.box(df, x = b_selected_col, y = 'diagnosis', title = chart_title, color = 'diagnosis'))
            else:
                st.plotly_chart(px.box(df, x = b_selected_col, title = chart_title))
                


    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("Visualizing Relationships")

        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        

        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {' '.join(selected_col_x.split('_')).title()} vs. {' '.join(selected_col_y.split('_')).title()}"

            if st.toggle("Diagnosis Hue on Scatterplot"):
                st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'diagnosis'))
            else: 
                st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title))

    # COUNTPLOTS
    if "Barplot" in eda_type:
        st.subheader("Visualizing Counts")

        c_selected_col = st.selectbox("Select your variable:", obj_cols, index = None)

        if c_selected_col:
            chart_title = f"Count of {' '.join(c_selected_col.split('_')).title()}"
            st.plotly_chart(px.bar(df['diagnosis']))

# MODELING

if page == "Modeling":
    st.title(":gear: Modeling")
    st.markdown("On this page, you can see how well different **machine learning models** make predictions on breast cancer diagnosis.")

    # Set up X and y
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean']
    X = df[features]
    y = df['diagnosis']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Model selection
    model_option = st.selectbox("Select a Model", ['KNN', 'Logistic Regression', 'Random Forest'], index = None)

    if model_option:
        # st.write(f"You selected {model_option}")

        if model_option == 'KNN':
            k_value = st.slider("Select the number of neighers (k)", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors = k_value)
        elif model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Random Forest':
            model = RandomForestClassifier()

        
        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)

            # Display Results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues')
            st.pyplot()


# PREDICTIONS

if page == "Make Predictions":
    st.title(":ribbon: Make Predictions on the Breast Cancer Dataset")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")

    r = st.slider("radius_mean", 0.0, 30.0, 0.0, 0.5)
    t = st.slider("texture_mean", 0.0, 40.0, 0.0, 0.5)
    p = st.slider("perimeter_mean", 40.0, 200.0, 0.0, 1.0)
    a = st.slider("area_mean", 0.0, 2500.0, 0.0, 5.0)
    smooth = st.slider("smoothness_mean", 0.0, 0.20, 0.0, 0.01)
    comp = st.slider("compactness_mean", 0.0, 0.35, 0.0, 0.01)
    conc = st.slider("concavity_mean", 0.0, 0.5, 0.0, 0.01)
    symm = st.slider("symmetry_mean", 0.0, 0.3, 0.0, 0.01)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'radius_mean': [r],
            'texture_mean': [t],
            'perimeter_mean': [p],
            'area_mean': [a],
            'smoothness_mean': [smooth],
            'compactness_mean': [comp],
            'concavity_mean': [conc],
            'symmetry_mean' : [symm]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean']
    X = df[features]
    y = df['diagnosis']

    # Model Selection
    model_option = st.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    if model_option:

        # Instantiating & fitting selected model
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        
        if st.button("Make a Prediction"):
            model.fit(X, y)
            prediction = model.predict(user_input)
            st.write(f"{model} predicts your tumor is {prediction[0]}")
            