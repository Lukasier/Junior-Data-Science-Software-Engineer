import pandas as pd

def clean_data(input_file, output_file):
    """
    Delete unnecessary features and NaN values from input file
    
    Parameters :
    
    input_file : str, path object or file-like object
    output_file : str or file handle
    
    """
    # Create data frame
    data = pd.read_csv(input_file, sep = ";")
    
    # Remove unnecessary features from data frame
    data = data.drop(["Name","Ticket","Cabin"], axis=1)
    
    # Remove NaN values from remaining features
    data = data.dropna()
    
    # Save ready-to-use file
    data.to_csv(output_file, index=False)
