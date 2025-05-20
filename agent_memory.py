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


def create_or_get_folder(folder_name, upload_to_google_drive=False):

    if upload_to_google_drive:

        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        results = drive_service.files().list(q=query).execute()
        items = results.get('files', [])

        if not items:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = drive_service.files().create(body=folder_metadata).execute()
            folder_id = folder['id']
        else:
            folder_id = items[0]['id']

    else:
        folder_path = os.path.join('./', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_id = folder_path

    return folder_id


def create_or_get_doc(folder_id, doc_name, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)
        docs_service = build('docs', 'v1', credentials=creds)

        query = f"name='{doc_name}' and '{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        items = results.get('files', [])

        if not items:
            doc_metadata = {
                'name': doc_name,
                'mimeType': 'application/vnd.google-apps.document',
                'parents': [folder_id]
            }
            doc = drive_service.files().create(body=doc_metadata).execute()
            document_id = doc['id']
        else:
            document_id = items[0]['id']

    else:
        file_path = os.path.join(folder_id, f'{doc_name}.md')
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('') 
        document_id = file_path

    return document_id


def get_file_content(file_id, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        service = build('drive', 'v3', credentials=creds)
        os.environ['SSL_VERSION'] = 'TLSv1_2'
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        content = request.execute()
        decoded_content = content.decode('utf-8')

    else:
        with open(file_id, 'r', encoding='utf-8') as file:
            decoded_content = file.read()
    return decoded_content


def append_content_in_doc(folder_id, doc_id, dict_list, upload_to_google_drive=False):

    json_string = json.dumps(dict_list, indent=4, ensure_ascii=False)

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)
        docs_service = build('docs', 'v1', credentials=creds)

        document = docs_service.documents().get(documentId=doc_id).execute()
        end_of_doc = document['body']['content'][-1]['endIndex'] - 1

        requests = [{
            'insertText': {
                'location': {'index': end_of_doc},
                'text': json_string + '\n\n'
            }
        }]
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

    else:
        with open(doc_id, 'a', encoding='utf-8') as file:
            file.write(json_string)


def clear_content_in_doc(doc_id, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        docs_service = build('docs', 'v1', credentials=creds)

        document = docs_service.documents().get(documentId=doc_id).execute()
        end_of_doc = document['body']['content'][-1]['endIndex'] - 1

        requests = [{
            'deleteContentRange': {
                'range': {
                    'startIndex': 1,
                    'endIndex': end_of_doc
                }
            }
        }]

        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

    else:
        with open(doc_id, 'w') as file:
            pass


def list_files_in_folder(folder_id, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        query = f"'{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])

        file_names = [file['name'] for file in files]

    else:
        file_names = [f for f in os.listdir(folder_id) if os.path.isfile(os.path.join(folder_id, f))]
    return file_names


def rename_doc_in_drive(folder_id, doc_id, new_name, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        update_request_body = {
            'name': new_name
        }

        update_response = drive_service.files().update(
            fileId=doc_id,
            body=update_request_body,
            fields='id,name'
        ).execute()

        update_name = update_response['name']

    else:

        directory, old_file_name = os.path.split(doc_id)
        extension = os.path.splitext(old_file_name)[1]

        new_file_name = new_name + extension
        new_file_path = os.path.join(directory, new_file_name)

        os.rename(doc_id, new_file_path)

        update_name = new_name

    return update_name


def delete_all_files_in_folder(folder_id, upload_to_google_drive=False):

    if upload_to_google_drive:
        creds = Credentials.from_authorized_user_file('token.json')
        drive_service = build('drive', 'v3', credentials=creds)

        query = f"'{folder_id}' in parents"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])

        for file in files:
            file_id = file['id']
            drive_service.files().delete(fileId=file_id).execute()

    else:
        for filename in os.listdir(folder_id):
            file_path = os.path.join(folder_id, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {repr(e)}')


class InterProject():

    def __init__(self,
                 project_name,
                 part_name,
                 folder_id=None,
                 doc_id=None,
                 doc_content=None,
                 upload_to_google_drive=False):

        self.project_name = project_name

        self.part_name = part_name

        self.upload_to_google_drive = upload_to_google_drive


        if folder_id == None:
            folder_id = create_or_get_folder(folder_name=project_name,
                                             upload_to_google_drive=upload_to_google_drive)
        self.folder_id = folder_id

        self.doc_list = list_files_in_folder(folder_id,
                                             upload_to_google_drive=upload_to_google_drive)

        if doc_id == None:
            doc_id = create_or_get_doc(folder_id=folder_id,
                                       doc_name=part_name,
                                       upload_to_google_drive=upload_to_google_drive)
        self.doc_id = doc_id

        self.doc_content = doc_content
        if doc_content != None:
            append_content_in_doc(folder_id=folder_id,
                                  doc_id=doc_id,
                                  qa_string=doc_content,
                                  upload_to_google_drive=upload_to_google_drive)

    def get_doc_content(self):
        self.doc_content = get_file_content(file_id=self.doc_id,
                                            upload_to_google_drive=self.upload_to_google_drive)

        return self.doc_content

    def append_doc_content(self, content):
        append_content_in_doc(folder_id=self.folder_id,
                              doc_id=self.doc_id,
                              dict_list=content,
                              upload_to_google_drive=self.upload_to_google_drive)

    def clear_content(self):
        clear_content_in_doc(doc_id=self.doc_id,
                             upload_to_google_drive=self.upload_to_google_drive)

    def delete_all_files(self):

        delete_all_files_in_folder(folder_id=self.folder_id,
                                   upload_to_google_drive=self.upload_to_google_drive)

    def update_doc_list(self):

        self.doc_list = list_files_in_folder(self.folder_id,
                                             upload_to_google_drive=self.upload_to_google_drive)

    def rename_doc(self, new_name):

        self.part_name = rename_doc_in_drive(folder_id=self.folder_id,
                                             doc_id=self.doc_id,
                                             new_name=new_name,
                                             upload_to_google_drive=self.upload_to_google_drive)


class ChatMessages():

    def __init__(self,
                 system_content_list=[],
                 question='Hello',
                 tokens_thr=None,
                 project=None):

        self.system_content_list = system_content_list
        system_messages = []
        history_messages = []
        messages_all = []
        system_content = ''
        history_content = question
        content_all = ''
        num_of_system_messages = 0
        all_tokens_count = 0

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if system_content_list != []:
            for content in system_content_list:
                system_messages.append({"role": "system", "content": content})

                system_content += content

            system_tokens_count = len(encoding.encode(system_content))
            messages_all += system_messages
            num_of_system_messages = len(system_content_list)

            if tokens_thr != None:
                if system_tokens_count >= tokens_thr:
                    print(
                        "If the number of tokens in system_messages exceeds the limit, the current system messages will not be entered into the model, readjust the number of external documents if necessary.")
                    system_messages = []
                    messages_all = []

                    num_of_system_messages = 0

                    system_tokens_count = 0

            all_tokens_count += system_tokens_count

        history_messages = [{"role": "user", "content": question}]

        messages_all += history_messages

        user_tokens_count = len(encoding.encode(question))

        all_tokens_count += user_tokens_count

        if tokens_thr != None:
            if all_tokens_count >= tokens_thr:
                print("The number of tokens for the current user issue exceeds the limit, the message cannot be entered into the model, please re-enter the user issue or adjust the number of external documents.")
                history_messages = []
                system_messages = []
                messages_all = []
                num_of_system_messages = 0
                all_tokens_count = 0

        self.messages = messages_all
        self.system_messages = system_messages
        self.history_messages = history_messages
        self.tokens_count = all_tokens_count
        self.num_of_system_messages = num_of_system_messages
        self.tokens_thr = tokens_thr
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.project = project

    def messages_pop(self, manual=False, index=None):
        def reduce_tokens(index):
            drop_message = self.history_messages.pop(index)
            self.tokens_count -= len(self.encoding.encode(str(drop_message)))

        if self.tokens_thr is not None:
            while self.tokens_count >= self.tokens_thr:
                reduce_tokens(-1)

        if manual:
            if index is None:
                reduce_tokens(-1)
            elif 0 <= index < len(self.history_messages) or index == -1:
                reduce_tokens(index)
            else:
                raise ValueError("Invalid index value: {}".format(index))

        self.messages = self.system_messages + self.history_messages

    def messages_append(self, new_messages):

        if type(new_messages) is dict or type(new_messages) is openai.openai_object.OpenAIObject:
            self.messages.append(new_messages)
            self.tokens_count += len(self.encoding.encode(str(new_messages)))

        elif isinstance(new_messages, ChatMessages):
            self.messages += new_messages.messages
            self.tokens_count += new_messages.tokens_count

        self.history_messages = self.messages[self.num_of_system_messages:]

        self.messages_pop()

    def copy(self):

        system_content_str_list = [message['content'] for message in self.system_messages]
        new_obj = ChatMessages(
            system_content_list=copy.deepcopy(system_content_str_list),
            question=self.history_messages[0]['content'] if self.history_messages else '',
            tokens_thr=self.tokens_thr
        )
        new_obj.history_messages = copy.deepcopy(self.history_messages)
        new_obj.messages = copy.deepcopy(self.messages)
        new_obj.tokens_count = self.tokens_count
        new_obj.num_of_system_messages = self.num_of_system_messages

        return new_obj

    def add_system_messages(self, new_system_content):
        system_content_list = self.system_content_list
        system_messages = []

        if type(new_system_content) == str:
            new_system_content = [new_system_content]

        system_content_list.extend(new_system_content)
        new_system_content_str = ''
        for content in new_system_content:
            new_system_content_str += content
        new_token_count = len(self.encoding.encode(str(new_system_content_str)))
        self.tokens_count += new_token_count
        self.system_content_list = system_content_list
        for message in system_content_list:
            system_messages.append({"role": "system", "content": message})
        self.system_messages = system_messages
        self.num_of_system_messages = len(system_content_list)
        self.messages = system_messages + self.history_messages

        self.messages_pop()


    def delete_system_messages(self):
        system_content_list = self.system_content_list
        if system_content_list != []:
            system_content_str = ''
            for content in system_content_list:
                system_content_str += content
            delete_token_count = len(self.encoding.encode(str(system_content_str)))
            self.tokens_count -= delete_token_count
            self.num_of_system_messages = 0
            self.system_content_list = []
            self.system_messages = []
            self.messages = self.history_messages

    def delete_function_messages(self):

        history_messages = self.history_messages

        for index in range(len(history_messages) - 1, -1, -1):
            message = history_messages[index]
            if message.get("function_call") or message.get("role") == "function":
                self.messages_pop(manual=True, index=index)

