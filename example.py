from contributions import get_contribution


# The function is capable of fitting
test1 = get_contribution(
    model=M,  # Xgboost model
    df=params,  # Dataframe with only the features
    calculate=True,  # Tells function to predict its own labels
    percentiles=10  # Tells function how many sampling points it wants
)

# If the
test2 = get_contribution(
    model=M,  # Xgboost model
    df=params,  # Dataframe with only the features
    target_df=target,  # Dataframe with only the labels
    calculate=False,  # Tells function not to predict its own labels
    percentiles=10  # Tells function how many sampling points it wants
)

# If the labels are already inside the feature dataframe
test3 = get_contribution(
    model=M,  # Xgboost model
    df=dataframe,  # Dataframe with features and labels
    target='TargetColumnName',  # The column where the labels are stored
    calculate=False,  # Tells function not to predict its own labels
    percentiles=10  # Tells function how many sampling points it wants
)