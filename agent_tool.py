import os
import openai
import copy
import glob
import shutil
from IPython.display import display, Code, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tiktoken

import numpy as np
import pandas as pd

import json
import io
import inspect
import requests
import re
import random
import string
import base64
import pymysql
import os.path
import matplotlib

from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseUpload
import base64
import email
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText
from openai.error import APIConnectionError

from bs4 import BeautifulSoup
import dateutil.parser as parser

import sys
from gptLearning import *

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from io import BytesIO

def function_to_call(available_functions, function_call_message):

    function_name = function_call_message["function_call"]["name"]

    fuction_to_call = available_functions.functions_dic[function_name]

    function_args = json.loads(function_call_message["function_call"]["arguments"])

    try:

        function_args['g'] = globals()

        function_response = fuction_to_call(**function_args)

    except Exception as e:
        function_response = "The function runs with the following error:" + repr(e)
        print(function_response)

    function_response_messages = {
        "role": "function",
        "name": function_name,
        "content": function_response,
    }

    return function_response_messages


def sql_inter(sql_query, g='globals()'):

    mysql_pw = os.getenv('MYSQL_PW')

    connection = pymysql.connect(
        host='localhost',
        user='root',
        passwd=mysql_pw,
        db='Ni_UiO66_Ce_db',
        charset='utf8'
    )

    try:
        with connection.cursor() as cursor:

            sql = sql_query
            cursor.execute(sql)

            results = cursor.fetchall()

    finally:
        connection.close()

    return json.dumps(results)


def extract_data(sql_query, df_name, g='globals()'):

    mysql_pw = os.getenv('MYSQL_PW')

    connection = pymysql.connect(
        host='localhost',
        user='root',
        passwd=mysql_pw,
        db='Ni_UiO66_Ce_db',
        charset='utf8'
    )

    g[df_name] = pd.read_sql(sql_query, connection)

    return "The %s variable creation has been completed successfully" % df_name


def python_inter(py_code, g='globals()'):

    global_vars_before = set(g.keys())
    try:
        exec(py_code, g)
    except Exception as e:
        return f"python_inter code execution error{repr(e)}"
    global_vars_after = set(g.keys())
    new_vars = global_vars_after - global_vars_before

    if new_vars:
        result = {var: g[var] for var in new_vars}
        return str(result)
    else:
        try:
            return str(eval(py_code, g))
        except Exception as e:
            try:
                exec(py_code, g)
                return "The code has been successfully executed"
            except Exception as e:
                pass
            return f"python_inter code execution error{repr(e)}"


def fig_inter(py_code, fname, g='globals()'):

    current_backend = matplotlib.get_backend()

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    local_vars = {"plt": plt, "pd": pd, "sns": sns}

    try:
        exec(py_code, g, local_vars)
    except Exception as e:
        return f"python_inter code execution error{repr(e)}"

    matplotlib.use(current_backend)

    fig = local_vars[fname]

    try:
        fig_url = upload_image_to_drive(fig)
        res = f"The code has been successfully run and the image created by the code has been saved to{fig_url}"

    except Exception as e:
        res = "Unable to upload images to Google Cloud Drive, please check the Google Cloud Drive folder ID and check the current network condition"

    print(res)
    return res


def upload_image_to_drive(figure, folder_id='...'):

    folder_id = folder_id
    creds = Credentials.from_authorized_user_file('token.json')
    drive_service = build('drive', 'v3', credentials=creds)

    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype='image/png', resumable=True)
    file_metadata = {
        'name': 'YourImageName.png',
        'parents': [folder_id],
        'mimeType': 'image/png'
    }
    image_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id,webContentLink'
    ).execute()

    return image_file["webContentLink"]


def auto_functions(functions_list):

    def functions_generate(functions_list):
        functions = []

        def chen_ming_algorithm(data):

            df_new = pd.read_json(data)
            res = np.sum(df_new, axis=1) - 1
            return res.to_json(orient='records')

        chen_ming_function_description = inspect.getdoc(chen_ming_algorithm)

        chen_ming_function_name = chen_ming_algorithm.__name__

        chen_ming_function = {"name": "chen_ming_algorithm",
                              "description": "A function used to execute Chen Ming's algorithm defines a special data set computation procedure",
                              "parameters": {"type": "object",
                                             "properties": {"data": {"type": "string",
                                                                     "description": "Execute Chen Ming's algorithm on the data set."},
                                                            },
                                             "required": ["data"],
                                             },
                              }

        for function in functions_list:
            function_description = inspect.getdoc(function)
            function_name = function.__name__

            user_message1 = 'Here is the function description for something: %s. ' % chen_ming_function_description + \
        'Given the function specification for this function, please create a Function object that describes the basic situation of this function. The function object is a dictionary in JSON format, \
        The dictionary has five requirements: \
        There are three key-value pairs in the dictionary. \
        In the first key-value pair, Key is the string name, and value is the name of the function: %s, also a string. \
        In the second key-value pair, Key is the string description, and value is the description of the function function, also a string. \
        In the third key-value pair, the Key is the string parameters, and value is a JSON Schema object that specifies the function parameter input specification. \
        The output must be a dictionary in JSON format, just the dictionary, without any preamble or explanation '% chen_ming_function_name

            assistant_message1 = json.dumps(chen_ming_function)

            user_prompt = 'Now we have another function named %s; The function description is: %s; \
                            Please help me create a function object for this function with a similar format. ' % (function_name, function_description)

            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "user", "name": "example_user", "content": user_message1},
                    {"role": "assistant", "name": "example_assistant", "content": assistant_message1},
                    {"role": "user", "name": "example_user", "content": user_prompt}]
            )
            functions.append(json.loads(response.choices[0].message['content']))
        return functions

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            functions = functions_generate(functions_list)
            break 
        except Exception as e:
            attempts += 1  
            print("An error occurred：", e)
            print("An error is reported due to the limit rate of the model. Pause for 1 minute and try again to invoke the model after 1 minute")
            time.sleep(60)

            if attempts == max_attempts:
                print("The maximum number of attempts has been reached and the program terminates.")
                raise
            else:
                print("Re-running...")
    return functions


class AvailableFunctions():

    def __init__(self, functions_list=[], functions=[], function_call="auto"):
        self.functions_list = functions_list
        self.functions = functions
        self.functions_dic = None
        self.function_call = None

        if functions_list != []:
            self.functions_dic = {func.__name__: func for func in functions_list}
            self.function_call = function_call
            if functions == []:
                self.functions = auto_functions(functions_list)

    def add_function(self, new_function, function_description=None, function_call_update=None):
        self.functions_list.append(new_function)
        self.functions_dic[new_function.__name__] = new_function
        if function_description == None:
            new_function_description = auto_functions([new_function])
            self.functions.append(new_function_description)
        else:
            self.functions.append(function_description)
        if function_call_update != None:
            self.function_call = function_call_update
