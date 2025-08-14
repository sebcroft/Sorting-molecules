# def nest_the_list(flat_list, n_cols=5):
#     # Calculate the number of rows
#     n_rows = (len(flat_list) + n_cols - 1) // n_cols  # Ceiling division
    
#     # Create the grid and labels
#     grid_list = []
#     labels = []
#     for row in range(n_rows):
#         start_idx = row * n_cols
#         end_idx = start_idx + n_cols
#         row_list = flat_list[start_idx:end_idx]
#         grid_list.append(row_list)
#     return grid_list 


def extract(data, index):
    """
    Recursively extract an item from tuples in a nested structure.

    Args:
        data (list or tuple): The nested list/tuple structure.
        index (int): The index of the item to extract from each tuple.

    Returns:
        list: The nested structure with the extracted item.
    """
    if isinstance(data, tuple):
        # If it's a tuple, return the desired item
        return data[index]
    elif isinstance(data, list):
        # If it's a list, process each element recursively
        return [extract(item, index) for item in data]
    else:
        # If it's not a list or tuple, return as is
        return data



def to_lowest_level(nested_list):
    """
    Extracts the lowest level of deep hierarchical list into a 1-level nested list i.e. [[...], [.], [..], [.....]]

    Args:
    nested_list : list
        A nested list of any depth
    Returns:
    list
        A nested list containing only lowest level/depth lists
    """
    # If the input is not a list, return an empty list (not a valid nested structure)
    if not isinstance(nested_list, list):
        return []

    # If the input contains no further lists, it is a lowest-level list
    if all(not isinstance(item, list) for item in nested_list):
        return [nested_list]

    # Otherwise, recursively process each element of the list
    result = []
    for item in nested_list:
        result.extend(to_lowest_level(item))
    return result