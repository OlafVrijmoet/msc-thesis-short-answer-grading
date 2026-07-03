
def prompt_user(prompt: str, user_options_values: dict):

    # get user options
    user_options = user_options_values.keys()

    while True:

        user_input = input(prompt).lower().strip()

        # try to convert user input to int
        try:
            user_input = int(user_input)
        except ValueError:
            user_input = user_input
        
        if user_input in user_options:

            return user_options_values[user_input]
            
        else:
            print("Invalid input!")
