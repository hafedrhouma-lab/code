import json
import pandas as pd


def create_message_chat_completion(system_message, user_input):
    message = [
        {'role': 'system',
         'content': system_message},

        {'role': 'user',
         'content': user_input.to_json()}
    ]
    return message


def create_requests_chat_completion(
        system_message: str,
        df_users_inputs: pd.DataFrame
):
    jobs = [
        {
            "model": "gpt-3.5-turbo",
            "messages": create_message_chat_completion(system_message, row)
        } for index, row in df_users_inputs.iterrows()
    ]

    data_to_write = ""
    for job in jobs:
        json_string = json.dumps(job)
        data_to_write += json_string + "\n"

    return data_to_write
