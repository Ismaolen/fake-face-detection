import inspect
import os


def print_status():
    frame = inspect.currentframe().f_back
    file_path = frame.f_code.co_filename
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    function_name = frame.f_code.co_name

    # Umwandeln des Funktionsnamens von snake_case zu Title Case
    function_name_readable = ' '.join(word.capitalize() for word in function_name.split('_'))

    line_number = frame.f_lineno

    CYAN = '\033[96m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    YELLOW = '\033[93m'

    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"{MAGENTA}┌───[File: {CYAN}{folder_name}/{file_name}{MAGENTA}]{RESET}")
    print(f"{MAGENTA}├── {BOLD}Funktion:{RESET} {function_name_readable}")
    print(f"{MAGENTA}├── {BOLD}Line:{RESET} {line_number}")
    print(f"{MAGENTA}└── {GREEN}✓ Status Okay{RESET}")
    print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")






