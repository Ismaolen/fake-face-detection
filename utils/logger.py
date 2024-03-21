import inspect
import os


def print_status():
    """
    Prints the status information including file, function, and line number.

    This function retrieves and prints the execution context of where it was called, 
    including the file name, function name, and line number. The function name is 
    converted from snake_case to Title Case. The output is color-coded for readability.

    Returns
    -------
    None
    """
    # Retrieve the calling frame to get context information
    frame = inspect.currentframe().f_back
    file_path = frame.f_code.co_filename  # Path of the file where the function was called
    folder_name = os.path.basename(os.path.dirname(file_path))  # Folder name of the file
    file_name = os.path.basename(file_path)  # Name of the file
    function_name = frame.f_code.co_name  # Name of the function where print_status was called

    # Convert function name from snake_case to Title Case for better readability
    function_name_readable = ' '.join(word.capitalize() for word in function_name.split('_'))

    line_number = frame.f_lineno  # Line number where print_status was called

    # Define ANSI color codes for colorful output
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    YELLOW = '\033[93m'

    # Print the status information in a structured and colorful format
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"{MAGENTA}┌───[File: {CYAN}{folder_name}/{file_name}{MAGENTA}]{RESET}")
    print(f"{MAGENTA}├── {BOLD}Funktion:{RESET} {function_name_readable}")
    print(f"{MAGENTA}├── {BOLD}Line:{RESET} {line_number}")
    print(f"{MAGENTA}└── {GREEN}✓ Status Okay{RESET}")
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")






