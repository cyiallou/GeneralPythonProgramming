'''
Password Validator is a program that validates passwords to match specific rules.

Rules:
 - Minimum length is 5
 - Maximum length is 10
 - Should contain at least one number
 - Should contain at least one special character (such as !, @, #, $, %, ^, &, etc
 - Should not contain spaces
 - Should contain at least 1 uppercase letter

Command line examples (all arguments are passed as strings):
1) Pass in a single password:   python3 script_name.py aBcd1234!@
2) Pass in a list of passwords: python3 script_name.py word1[\'F wor\'d2]@ asd 
   (use whitespace as separator and \ to pass special characters e.g. \')
'''

import sys
from string import punctuation, digits, ascii_uppercase


def check_password_validity(password):
    '''
    Returns a boolean list with True if password input is valid or
    False otherwise.

    Parameters
    ----------
    password: list of strings
    '''
    if type(password)==list:  # executes only once
        n_items = len(password)
        validity_flag_arr = [
                check_password_validity(str(password[i])) for i in range(n_items)
                ]
        return validity_flag_arr

    validity_flag = False  # initialise to flag an invalid password
    # first check for length and white space
    if (len(password)<5) | (len(password)>10) | (' ' in password):
        return validity_flag
    else:
        # check for numbers
        for d in digits:
            if d in password:
                for i in punctuation:
                    if i in password:
                        for k in ascii_uppercase:
                            if k in password:
                                validity_flag = True
                                break

    return validity_flag


if __name__=='__main__':
    if len(sys.argv)==1:
        password_in = input(
                'Enter as many passwords as you want to '
                'check (separate them with whitespace): '
                ).split()
    else:
        password_in = sys.argv[1:]

    validity_flag_arr = check_password_validity(password=password_in)

    # Give feedback to the user
    invalid_indices = [i for i,x in enumerate(validity_flag_arr) if not x]
    n_passwords_to_check = len(password_in)
    if len(invalid_indices) > 0:
        if n_passwords_to_check > 1:
            print(f'Passwords with indices {invalid_indices} are invalid')
        else:
            print('Password is invalid')
    else:
        if n_passwords_to_check>1:
            print('All submitted passwords are valid')
        else:
            print('Password is valid')

