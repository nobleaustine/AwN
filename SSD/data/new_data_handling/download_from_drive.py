# pip install --upgrade google-auth google-auth-httplib2 google-auth-oauthlib google-api-python-clientvvv
# AHFP-Scanrunde-1 1KLlu6_3XdpfMPg9fiJwPY3DDLXQWmVhW
# AHFP-Scanrunde-2 1GZ7rrQX8W228NBEvDV8oB6q9K47EkDdH
# AHFP-Scanrunde-3 13teMo95gNI_b_vkhEMZfvChTmUTQ0AMV
# AHFP-Scanrunde-4 18Qk_KlvDhuJZp4EUrSzzZ7WbMspb7kD2
# AHFP-Scanrunde-5 1sAyFTBzDDMEclhCFeNL8QYIypRisOyOZ
# AHFP-Scanrunde-6 1rfRKPbszS4HlQUnxF_AvlOTt8DNpSYkY
# Contrast - non contrast dataset.docx 146XG5vIpsH5RxEBlktFbMDPxEcEQEFo1


from __future__ import print_function
import os
import io
import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

SERVICE_ACCOUNT_FILE = './credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    """Authenticate using the service account and return the Drive service."""
    
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def download_file(service, file_id, file_name,file_type, save_path):
    """ Download a file from Google Drive """

    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(save_path, file_name)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False

    with open('new_data_paths.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([save_path.split('new_data/')[1] + f'/{file_name}.{file_type}'])
    
    while not done:
        status, done = downloader.next_chunk()
        print(f'\r{file_name}({file_type}):{int(status.progress() * 100)}%.', end='', flush=True)
    print()

def download_folder(service, folder_id, save_path):
    """Download a folder and its contents from Google Drive."""

    try :
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f'Downloading folder: {save_path.split("new_data/")[1]}')

        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='nextPageToken, files(id, name, mimeType)').execute()
        items = results.get('files', [])

        if not items:
            print('No files found in the folder.')
        else:
            for item in items:
                file_id = item['id']
                file_name = item['name']
                mime_type = item['mimeType']
                if mime_type == 'application/vnd.google-apps.folder':
                    new_folder_path = os.path.join(save_path, file_name)
                    download_folder(service, file_id, new_folder_path)
                elif mime_type == 'application/dicom' or file_name.endswith('.dcm') or file_name.endswith('.nii.gz') or file_name.endswith('.txt'):
                    download_file(service, file_id, file_name, mime_type, save_path)
                else:
                    with open('error_data_paths.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([f'Skipping {save_path}/{file_name} ({mime_type})'])
    except HttpError as e:
        print(f'An error occurred: {e}')

def main():
    """Authenticate and download a folder from Google Drive."""
    ids = ["1sAyFTBzDDMEclhCFeNL8QYIypRisOyOZ",
           "1rfRKPbszS4HlQUnxF_AvlOTt8DNpSYkY",
            "146XG5vIpsH5RxEBlktFbMDPxEcEQEFo1"]
    for id in ids:
        save_path = '/cluster/home/austinen/NTNU/DATA/new_data/'

        try :
            service = authenticate()
            metadata = service.files().get(fileId=id).execute()
            mime_type = metadata['mimeType']
            file_name = metadata['name']

            # The initial path is a file, download it
            if mime_type != 'application/vnd.google-apps.folder':
                download_file(service, id, file_name, mime_type, save_path)
            else:
            # the intial path is a folder, download its contents
                new_path = os.path.join(save_path, file_name)
                download_folder(service, id, new_path)

        except HttpError as e:
            print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
