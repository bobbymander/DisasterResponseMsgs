import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

# load in the nltk dictionaries we need
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """Load the previously saved data into a dataframe.

    Parameters:
    database_filepath (str):  The path of the database file that we should read from.

    Returns:
    X (DataFrame): The X data to be used for our model, the message column
    Y (DataFrame):  The Y data to be used for our model, the category columns
    categories (list(str)):  The list of categories represented by the Y data

    """
    # get the data from the sql file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('cleanmsgs', engine)
    
    # The message column forms our X data
    X = df['message']
    
    # The category columns form our Y data
    Y = df.iloc[:, 3:]
    
    # Return all 3 pieces
    return X, Y, list(Y)

def tokenize(text):
    """Extract the tokens from the text string provided.

    Parameters:
    text (str): Text message to tokenize

    Returns:
    tokens (array[str]): Array of token strings extracted from text

    """
    # convert to lowercase and remove anything that's not a letter or number
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # find root words and save only those
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Build a model for use to classify the messages into the corresponding categories.

    Parameters:
    None
    
    Returns:
    model (estimator): The model created for message classification.

    """
    # create a pipeline using our familiar NLP friends along with a MOC since we have so many categories
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('moc', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # create parameters for use with grid search
    # keep it simple as grid search takes quite a long time!
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'moc__estimator__n_estimators': [10, 20],
    }

    # the grid search produces the final model
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Show the prediction metrics for the model for each category

    Parameters:
    model (estimator): The model created for message classification.
    X_test (DataFrame): The X data to be used for testing our model, the message column
    Y_test (DataFrame):  The Y data to be used for testing our model, the category columns
    category_names (list(str)):  The list of categories represented by the Y data

    Returns:
    None
    """
    # perform the predictions
    y_pred = model.predict(X_test)
    
    # print out the results using the classification report for each category
    colnum = 0
    for col in category_names:
        report = classification_report(Y_test[[col]], y_pred[:, colnum])
        colnum += 1
        print(col, report)

def save_model(model, model_filepath):
    """Save the final model to a file for use later

    Parameters:
    model (estimator): The model created for message classification.
    model_filepath (str):  The path of the database file that we should write the model to.

    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """Create, train, and save a model for use in message classification

    Parameters:
    None
    
    Returns:
    None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()