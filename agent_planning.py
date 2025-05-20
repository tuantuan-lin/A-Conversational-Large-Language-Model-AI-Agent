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



def modify_prompt(messages, action='add', enable_md_output=True, enable_COT=True):

    cot_prompt = "Think step by step and come to a conclusion."

    md_prompt = "Output any answers in markdown format."

    if action == 'add':
        if enable_COT:
            messages.messages[-1]["content"] += cot_prompt
            messages.history_messages[-1]["content"] += cot_prompt

        if enable_md_output:
            messages.messages[-1]["content"] += md_prompt
            messages.history_messages[-1]["content"] += md_prompt

    elif action == 'remove':
        if enable_md_output:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(md_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(md_prompt, "")

        if enable_COT:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(cot_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(cot_prompt, "")

    return messages


def get_gpt_response(model,
                     messages,
                     available_functions=None,
                     is_developer_mode=False,
                     is_enhanced_mode=False):

    time.sleep(60)

    if is_developer_mode:
        messages = modify_prompt(messages, action='add')

    while True:
        try:

            if available_functions == None:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages)

            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages,
                    functions=available_functions.functions,
                    function_call=available_functions.function_call
                )
            break

        except APIConnectionError as e:

            if is_enhanced_mode:

                msg_temp = messages.copy()

                question = msg_temp.messages[-1]["content"]

                new_prompt = "Here's what the user asked: %s. The problem is somewhat complex and the user's intentions are not clear. \
                    Write a paragraph that directs the user to ask a new question." % question

                try:
                    msg_temp.messages[-1]["content"] = new_prompt
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=msg_temp.messages)

                    display(Markdown(response["choices"][0]["message"]["content"]))

                    user_input = input("Please re-enter the question, enter exit to exit the current conversation")
                    if user_input == "exit":
                        print("The current model cannot return results and has exited.")
                        return None
                    else:
                        messages.history_messages[-1]["content"] = user_input
                        time.sleep(60)

                        response_message = get_gpt_response(model=model,
                                                            messages=messages,
                                                            available_functions=available_functions,
                                                            is_developer_mode=is_developer_mode,
                                                            is_enhanced_mode=is_enhanced_mode)

                        return response_message

    if is_developer_mode:
        messages = modify_prompt(messages, action='remove')

    return response["choices"][0]["message"]


def get_chat_response(model,
                      messages,
                      available_functions=None,
                      is_developer_mode=False,
                      is_enhanced_mode=False,
                      delete_some_messages=False,
                      is_task_decomposition=False):

    if not is_task_decomposition:

        response_message = get_gpt_response(model=model,
                                            messages=messages,
                                            available_functions=available_functions,
                                            is_developer_mode=is_developer_mode,
                                            is_enhanced_mode=is_enhanced_mode)

    if is_task_decomposition or (is_enhanced_mode and response_message.get("function_call")):

        is_task_decomposition = True
        task_decomp_few_shot = add_task_decomposition_prompt(messages)
        response_message = get_gpt_response(model=model,
                                            messages=task_decomp_few_shot,
                                            available_functions=available_functions,
                                            is_developer_mode=is_developer_mode,
                                            is_enhanced_mode=is_enhanced_mode)
        if response_message.get("function_call"):
            print("The current task does not need to be disassembled and can be run directly.")

    if delete_some_messages:
        for i in range(delete_some_messages):
            messages.messages_pop(manual=True, index=-1)

    if not response_message.get("function_call"):

        text_answer_message = response_message

        messages = is_text_response_valid(model=model,
                                          messages=messages,
                                          text_answer_message=text_answer_message,
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=is_enhanced_mode,
                                          delete_some_messages=delete_some_messages,
                                          is_task_decomposition=is_task_decomposition)

    elif response_message.get("function_call"):

        function_call_message = response_message

        messages = is_code_response_valid(model=model,
                                          messages=messages,
                                          function_call_message=function_call_message,
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=is_enhanced_mode,
                                          delete_some_messages=delete_some_messages)

    return messages

def is_code_response_valid(model,
                           messages,
                           function_call_message,
                           available_functions=None,
                           is_developer_mode=False,
                           is_enhanced_mode=False,
                           delete_some_messages=False):

    code_json_str = function_call_message["function_call"]["arguments"]

    try:
        code_dict = json.loads(code_json_str)
    except Exception as e:
        print("json character parsing error, re-creating code...")

        messages = get_chat_response(model=model,
                                     messages=messages,
                                     available_functions=available_functions,
                                     is_developer_mode=is_developer_mode,
                                     is_enhanced_mode=is_enhanced_mode,
                                     delete_some_messages=delete_some_messages)

        return messages

    def convert_to_markdown(code, language):
        return f"```{language}\n{code}\n```"

    if code_dict.get('sql_query'):
        code = code_dict['sql_query']
        markdown_code = convert_to_markdown(code, 'sql')

    elif code_dict.get('py_code'):
        code = code_dict['py_code']
        markdown_code = convert_to_markdown(code, 'python')

    else:
        markdown_code = code_dict

    display(Markdown(markdown_code))

    if is_developer_mode:
        user_input = input("Run the code directly (1) or feed back changes and let the model make changes to the code before running (2)")
        if user_input == '1':
            print("Ok, running code, please wait...")

        else:
            modify_input = input("Ok, please enter modification suggestion:")

            messages.messages_append(function_call_message)
            messages.messages_append({"role": "user", "content": modify_input})

            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode,
                                         delete_some_messages=2)

            return messages

    function_response_message = function_to_call(available_functions=available_functions,
                                                 function_call_message=function_call_message)

    messages = check_get_final_function_response(model=model,
                                                 messages=messages,
                                                 function_call_message=function_call_message,
                                                 function_response_message=function_response_message,
                                                 available_functions=available_functions,
                                                 is_developer_mode=is_developer_mode,
                                                 is_enhanced_mode=is_enhanced_mode,
                                                 delete_some_messages=delete_some_messages)

    return messages

def check_get_final_function_response(model,
                                      messages,
                                      function_call_message,
                                      function_response_message,
                                      available_functions=None,
                                      is_developer_mode=False,
                                      is_enhanced_mode=False,
                                      delete_some_messages=False):

    fun_res_content = function_response_message["content"]

    if "error" in fun_res_content:

        print(fun_res_content)

        if not is_enhanced_mode:
            display(Markdown("**Efficient Debug Agent is being instantiated. Efficient Debug Agent is about to be executed...**"))
            debug_prompt_list = ['The code you wrote reported an error, please modify the code according to the error message and re-execute.']

        else:
          
            display(Markdown(
                "** Deep debug is about to be executed. This debug process will automatically execute multiple sessions. Please wait patiently. Being instantiated Deep Debug Agent...**"))
            display(Markdown("**deep debug Agent...**"))
            debug_prompt_list = ["The code executed before reported an error, where do you think the code was written wrong?",
                                 "Well. So according to your analysis, in order to solve this error, theoretically, how should it be done?",
                                 "Very good, then please follow your logic to write the corresponding code and run."]


        msg_debug = messages.copy()

        msg_debug.messages_append(function_call_message)

        msg_debug.messages_append(function_response_message)

        for debug_prompt in debug_prompt_list:
            msg_debug.messages_append({"role": "user", "content": debug_prompt})
            display(Markdown("**From Debug Agent:**"))
            display(Markdown(debug_prompt))

            display(Markdown("**From MOFGen:**"))
            msg_debug = get_chat_response(model=model,
                                          messages=msg_debug,
                                          available_functions=available_functions,
                                          is_developer_mode=is_developer_mode,
                                          is_enhanced_mode=False,
                                          delete_some_messages=delete_some_messages)

        messages = msg_debug.copy()

    else:
        time.sleep(60)
        messages.messages_append(function_call_message)
        messages.messages_append(function_response_message)
        messages = get_chat_response(model=model,
                                     messages=messages,
                                     available_functions=available_functions,
                                     is_developer_mode=is_developer_mode,
                                     is_enhanced_mode=is_enhanced_mode,
                                     delete_some_messages=delete_some_messages)

    return messages


def is_text_response_valid(model,
                           messages,
                           text_answer_message,
                           available_functions=None,
                           is_developer_mode=False,
                           is_enhanced_mode=False,
                           delete_some_messages=False,
                           is_task_decomposition=False):

    answer_content = text_answer_message["content"]

    print("\n")
    display(Markdown(answer_content))

    user_input = None

    if not is_task_decomposition and is_developer_mode:
        user_input = input("Please ask whether to execute task (1) according to this process, \
Or suggest changes to the current execution process (2), \
Or rerun the question (3), \
Or simply exit the conversation (4)")
        if user_input == '1':
            messages.messages_append(text_answer_message)
            print("The results of this session have been saved")

    elif is_task_decomposition:
        user_input = input("Please ask whether to execute task (1) according to this process, \
Or suggest changes to the current execution process (2), \
Or rerun the question (3), \
Or simply exit the conversation (4)")
        if user_input == '1':
            messages.messages_append(text_answer_message)
            print("Ok, the above process will be implemented step by step")
            messages.messages_append({"role": "user", "content": "Very good, follow this process step by step。"})
            is_task_decomposition = False
            is_enhanced_mode = False
            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode,
                                         delete_some_messages=delete_some_messages,
                                         is_task_decomposition=is_task_decomposition)

    if user_input != None:
        if user_input == '1':
            pass
        elif user_input == '2':
            new_user_content = input("Ok, enter suggestions for changes to the model results:")
            print("Okay, it's being modified.")

            messages.messages_append(text_answer_message)

            messages.messages_append({"role": "user", "content": new_user_content})

            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode,
                                         delete_some_messages=2,
                                         is_task_decomposition=is_task_decomposition)

        elif user_input == '3':
            new_user_content = input("Ok, please rephrase the question:")

            messages.messages[-1]["content"] = new_user_content

            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         is_developer_mode=is_developer_mode,
                                         is_enhanced_mode=is_enhanced_mode,
                                         delete_some_messages=delete_some_messages,
                                         is_task_decomposition=is_task_decomposition)

        else:
            print("Ok, you are out of the current conversation")

    else:
        messages.messages_append(text_answer_message)

    return messages
