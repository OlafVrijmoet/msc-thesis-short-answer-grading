
# extracts the relevent information from the datasets
def retrieve_relevant_columns(
    df,
    df_name,
    
    row_id,
    question,
    question_id,
    student_answer,
    reference_answer,
    assigned_points,
    max_points,
    domain
):
    
    relevant_data = [
        row_id,
        question,
        question_id,
        student_answer,
        reference_answer,
        assigned_points,
        max_points,
        domain
    ]
    column_names = [data.name for data in relevant_data if data.name != "" and data.column]
    
    # get relevant columns
    standardised = df.loc[:, column_names]
    
    # rename columns
    rename_columns = [{"default_name": data.default_name, "name": data.name} for data in relevant_data if data.column == True]
    rename_columns = {data["name"]: data["default_name"] for data in rename_columns}
    standardised = standardised.rename(columns=rename_columns)
    
    # add dummy variable to empty columns
    add_columns = [{"name": data.name, "value": data.value} for data in relevant_data if data.column == False]
    add_columns = {data["name"]: data["value"] for data in add_columns}
    
    standardised = standardised.assign(**add_columns)
    
    return standardised
