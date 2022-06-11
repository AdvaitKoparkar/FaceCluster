import os
import pickle as pkl
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow, InstalledAppFlow

PHOTOS_API_NAME = 'photoslibrary'
PHOTOS_API_VERSION = 'v1'
PHOTOS_SCOPE = ['https://www.googleapis.com/auth/photoslibrary',
                'https://www.googleapis.com/auth/photoslibrary.sharing']
PHOTOS_CRED_FOLDER = os.path.join('tokens', 'creds')
PHOTOS_CLIENT_FOLDER = os.path.join('tokens', 'clients')

PHOTOS_CLIENT_IDS = {
    'ak': {
        'name': 'client_secret_842145595125-6ufefnpt4ctlclms5ga97bo3hku4pah1.apps.googleusercontent.com'
    },
}

def photosCreateService(clienID):
    cred = _getCred(clienID)
    service = _buildService(clienID, cred)
    return service

def _buildService(clientID, cred):
    service = build(PHOTOS_API_NAME, PHOTOS_API_VERSION, credentials=cred, static_discovery=False)
    return service

def _getCred(clientID):
    cred = None
    savedCredFname = _getCredFname(clientID)
    if not os.path.isfile(savedCredFname):
        secretFname = _getSecretFname(clientID)
        flow = InstalledAppFlow.from_client_secrets_file(secretFname, PHOTOS_SCOPE)
        cred = flow.run_local_server()
        with open(savedCredFname, 'wb') as token:
            pkl.dump(cred, token)
    else:
        with open(savedCredFname, 'rb') as token:
            cred = pkl.load(token)
    if not cred.valid:
        cred.refresh(Request())
    return cred

def _getCredFname(clienID):
    name = PHOTOS_CLIENT_IDS[clienID]['name']
    credFname = os.path.join(PHOTOS_CRED_FOLDER, f'{name}.pkl')
    return credFname

def _getSecretFname(clienID):
    name = PHOTOS_CLIENT_IDS[clienID]['name']
    secretFname = os.path.join(PHOTOS_CLIENT_FOLDER, f'{name}.json')
    return secretFname
