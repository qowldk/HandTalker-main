import numpy as np

def check_last_number_in_frames(npy_file_path):
    try:
        # Load the numpy array from the .npy file
        array = np.load(npy_file_path, allow_pickle=True)
        
        # Print the shape of the loaded array for debugging
        print("Loaded array shape:", array.shape)
        
        # Ensure the array is 2-dimensional (frames x elements)
        if array.ndim != 2:
            raise ValueError("The loaded numpy array is not 2-dimensional")
        
        # Iterate over each frame and print the last element
        for i, frame in enumerate(array):
            if frame.size == 0:
                print(f"Frame {i} is empty")
            else:
                # Check if the last element is numeric
                last_number = frame[-1]
                if isinstance(last_number, (int, float)):
                    print(f"Frame {i} last number: {last_number}")
                else:
                    print(f"Frame {i} last element is not a number: {last_number}")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError:
        print(f"File not found: {npy_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the correct file path using raw string or forward slashes
npy_file_path = r'C:\Users\dmb56\Desktop\HandTalker\ML\signlang\dataset\0_안녕하세요_s_0.npy'
check_last_number_in_frames(npy_file_path)
