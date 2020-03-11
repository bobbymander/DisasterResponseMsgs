import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from csv files for messages and categories.

    Parameters:
    messages_filepath (str): The path to the messages csv file
    categories_filepath (str): The path to the categories csv file

    Returns:
    df (DataFrame): The dataframe containing the combination of message and category data.

    """
    messages = pd.read_csv(messages_filepath)
    messages.drop('original', axis=1, inplace=True)
    categories = pd.read_csv(categories_filepath)
    
    # merge the messages and categories using the id column
    df = messages.merge(categories, on = 'id', how = 'inner', validate = 'many_to_many')
    
    return df

def clean_data(df):
    """Clean the merged data obtained for messages and categories

    Parameters:
    df (DataFrame): The dataframe containing the combination of message and category data.

    Returns:
    df (DataFrame): The dataframe containing the cleaned combination of message and category data.

    """
    # split categories into the individual components
    categories = df['categories'].str.split(';', expand=True)
    
    # looking at the first row, remove all suffixes from the category names and rename the columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert each column's data to the actual value for the category (0 or 1)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # do some final cleanup by dropping unneeded columns, duplicates, and NaN values    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """Save a dataframe to a database file

    Parameters:
    df (DataFrame): The dataframe containing the combination of message and category data.
    database_filename (str):  The path of the database that we should create using the data.
    
    Returns:
    None
    """
    # create an sql engine and save the data to the db file
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('cleanmsgs', engine, index=False, if_exists='replace')

def main():
    """Process the data for messages and categories so that it is clean and is saved for later use.

    Parameters:
    None
    
    Returns:
    None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()