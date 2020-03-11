# Disaster Response Pipeline Project

The purpose of this project is to analyze text messages from people who are involved with a disaster.

Based on this NLP text analysis, we will determine what labels to apply to the message so that 
emergency personnel can act on them as needed.  We will use NLP techniques along with machine learning
classification using pipelines and grid search to accomplish this.

## Data
The data for this project can be found below.  There are 2 datasets, one for dogs and one for human faces.

Messages:	https://github.com/bobbymander/DisasterResponseMsgs/blob/master/data/disaster_messages.csv
Categories:	https://github.com/bobbymander/DisasterResponseMsgs/blob/master/data/disaster_categories.csv

## Files

app/run.py:  Used to run the Flask web app along with go.html and master.html
data/process_data.py:  Used to load and process the datasets described above
models/train_classifier.py:  Used to load the cleaned data, train a model with it using NLP and machine learning, and to evaluate the results

## Setup
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/

## Libraries

A listing of libraries installed and used by the notebook is below:

1.  numpy:  computation
2.  pandas:  data manipulation
3.  sqlalchemy:  storing and loading data into an SQL DB file
4.  nltk:  NLP tokenizing libraries
5.  sklearn:  machine learning classification libraries
6.  pickle:  model storing/loading libraries