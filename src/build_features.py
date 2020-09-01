import pandas as pd
import sklearn

def create_valid_features(input_file, output_file, force_write = True):
    """
    Changes cateogrical data from dataset into numberic values
    
    Parameters:
    
    input_file : str, path object or file-like object
    output_file : str or file handle
    
    """
    # Create data frame
    df = pd.read_csv(input_file)

    # Create numeric representation of categorical features (sex,embarked)
    # I took "sklearn" as a suggestion to use LabelEncoder :)
    lb_make = LabelEncoder()
    df["Sex"] = lb_make.fit_transform(df["Sex"])
    df["Embarked"] = lb_make.fit_transform(df["Embarked"])
    
    # Create feature desribing wheter person was alone or not based on family size
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df.loc[df["FamilySize"] > 0, "IsAlone"] = 0
    df.loc[df["FamilySize"] == 0, "IsAlone"] = 1

    df.to_csv(output_file, index=False)