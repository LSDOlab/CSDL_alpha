

def print_tabularized(
        table_keys: list[str],
        table_rows: list[list[str, float]],
        title = None,
        csv:str = None,
        ):
    """Prints a table of values in a tabular format.

    Parameters
    ----------
    table_keys : list[str]
        Strings representing the column headers
    table_rows : list[list[str, float]]
        List of lists of strings and floats representing the rows of the table
    title : str, optional
        Title of the table, by default None
    csv : str, optional
        If provided, writes the table to a csv file, by default None
    """

    # get the maximum length of each column
    buffer = 2
    column_lengths = [len(key) for key in table_keys]
    for row in table_rows:
        if len(row) != len(table_keys):
            raise ValueError("Row length does not match number of keys")
        for i, value in enumerate(row):
            column_lengths[i] = max(column_lengths[i], len(str(value)))    
    total_length = sum(column_lengths) + buffer*len(column_lengths)

    # print the table:
    # title
    if title is not None:
        print(f'\n{title}\n' + '-'*len(title))
    
    # header
    for i, key in enumerate(table_keys):
        print(f"{key:<{column_lengths[i]+buffer}}", end=" ")    
    print('\n'+'-'*total_length)
    
    # data rows
    for row in table_rows:
        for i, value in enumerate(row):
            print(f"{value:<{column_lengths[i]+buffer}}", end=" ")
        print()

    # write to csv
    if csv is not None:
        if not isinstance(csv, str):
            raise ValueError("csv aegument must be a string")
        with open(csv, 'w') as f:
            f.write(",".join(table_keys) + "\n")
            for row in table_rows:
                f.write(",".join([str(value) for value in row]) + "\n")



