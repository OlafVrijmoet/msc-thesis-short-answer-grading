
import numpy as np

def str_to_array(s):
    """
    Converts a string representation of a numpy array back to a numpy array.
    
    Parameters:
    s (str): String representation of a numpy array.
    
    Returns:
    np.ndarray: The string converted back to a numpy array.
    """
    # Split the shape and the data from the string
    shape_str, data_str = s.split("|", 1)

    # Convert the shape string back to a tuple of ints
    shape = tuple(map(int, shape_str.strip('()').split(',')))

    # Convert the data string back to bytes
    bytes_data = bytes.fromhex(data_str)

    # Convert the bytes back to a numpy array
    array = np.frombuffer(bytes_data, dtype=np.float32)

    # Reshape the array to its original shape
    array = array.reshape(shape)
    
    return array

def array_to_str(array):
    """
    Converts a numpy array to a string representation.
    
    Parameters:
    array (np.ndarray): The numpy array to be converted.
    
    Returns:
    str: String representation of the numpy array.
    """
    # Convert the numpy array to bytes
    bytes_data = array.tobytes()

    # Convert the bytes to a string
    data_str = bytes_data.hex()

    # Convert the shape of the array to a string
    shape_str = str(array.shape)

    # Combine the shape string and the data string into one string
    s = shape_str + "|" + data_str

    return s
