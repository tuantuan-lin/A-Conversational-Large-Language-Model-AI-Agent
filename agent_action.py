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


class MOFGen():
    def __init__(self,
                 api_key,
                 model='gpt-4o',
                 system_content_list=[],
                 project=None,
                 messages=None,
                 available_functions=None,
                 is_enhanced_mode=False,
                 is_developer_mode=False):

        self.api_key = api_key
        self.model = model
        self.project = project
        self.system_content_list = system_content_list
        tokens_thr = None

        if '1106' in model:
            tokens_thr = 110000
        elif 'gpt-4' in model:
            tokens_thr = 124000
        elif '16k' in model:
            tokens_thr = 12000
        elif '4-0613' in model:
            tokens_thr = 7000
        else:
            tokens_thr = 3000

        self.tokens_thr = tokens_thr

        self.messages = ChatMessages(system_content_list=system_content_list,
                                     tokens_thr=tokens_thr)

        if messages != None:
            self.messages.messages_append(messages)

        self.available_functions = available_functions
        self.is_enhanced_mode = is_enhanced_mode
        self.is_developer_mode = is_developer_mode

    def chat(self, question=None):

        head_str = "▌ Model set to %s" % self.model
        display(Markdown(head_str))

        if question != None:
            self.messages.messages_append({"role": "user", "content": question})
            self.messages = get_chat_response(model=self.model,
                                              messages=self.messages,
                                              available_functions=self.available_functions,
                                              is_developer_mode=self.is_developer_mode,
                                              is_enhanced_mode=self.is_enhanced_mode)

        else:
            while True:
                self.messages = get_chat_response(model=self.model,
                                                  messages=self.messages,
                                                  available_functions=self.available_functions,
                                                  is_developer_mode=self.is_developer_mode,
                                                  is_enhanced_mode=self.is_enhanced_mode)

                user_input = input("Do you have any other questions? (Enter exit to end the conversation): ")
                if user_input == "exit":
                    break
                else:
                    self.messages.messages_append({"role": "user", "content": user_input})

    def reset(self):

        self.messages = ChatMessages(system_content_list=self.system_content_list)

    def upload_messages(self):

        if self.project == None:
            print("You need to enter the project parameter (which needs to be an InterProject object) before you can upload messages")
            return None
        else:
            self.project.append_doc_content(content=self.messages.history_messages)
