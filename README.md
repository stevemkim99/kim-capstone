# Final Capstone Project

My capstone project for Coding Temple-Data Analyst.

In this project I performed data cleaning, exploratory data analysis (EDA) using Plotly, built predictive models using KNN, Random Forest, and Logistic Regression, and have showcased my results through a Streamlit app.


## Dataset

The dataset used in this project is the "Breast Cancer Dataset", which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset). The dataset contains information about patient's tumor sizing, including features such as area, radius, symmetry, and more.


## Tasks
In this project, my primary task is to read the breast cancer dataset into a Jupyter Notebook that I will create. Begin by performing essential data cleaning tasks to ensure the dataset is ready for analysis. Steps taken to clean the data were checking to see if there were any null values in the dataset or checking for any outliers. Once checking for these in the dataset it was ready to create machine learning models. Utilize Plotly for initial exploratory data analysis (EDA), creating interactive visualizations to gain insights into the dataset. Subsequently, employ machine learning techniques, including K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest, to build predictive models. Evaluate and compare the performance of these models to determine their effectiveness in predicting breast cancer diagnosis. Lastly, present my findings in a Streamlit app, providing an interactive and user-friendly platform to showcase the results of my analysis and model performance. 

### Creating a Baseline Model
First we created a baseline model, which essentially is just the diagnosis of all the benign tumors in the data. This makes the most "basic" prediction for the tumor diagnosis, it will not be accurate but we can use it as a baseline of whether or not our regression model will be closer or further away from predicting a better or worse sale price. The baseline model scored a 62.7%, meaning if we were to diagnosis every tumor as benign we would be accurate 62.7% of the time.

## Creating Machine Learning Models
Once we have created a baseline models, we can use machine learning to see if these models will perform better than our baseline model. We used StandardScaler, KNN(K Nearest Neighbors), LogisticRegression, and RandomForest. LogisticRegression scored the best testing accuracy with a whopping score of 97.9%. This is extremely high and much better than our baseline model, there is only a 2.1% chance of error with the LogisticRegression model. StandardScaler, KNN, and RandomForest still all scored high with scores of 95.8%, 97.2%, and 97.2% respectively. 


## Conclusion
Based off of these findings, there are many many many factors that go into the final diagnosis. The bigger the size of the tumor the more likely the tumor will be malignant. If you or someone you know is needing medical attention please get professional help, do not rely on these models to diagnosis. 



