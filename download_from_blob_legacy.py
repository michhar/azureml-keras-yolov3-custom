"""
Download data from blob storage with legacy package version (tested with legacy
Python Azure Blob SDK v2.1)

Overwrites any existing folder from this container on local system.

Make sure to set the environment variables before running:
STORAGE_ACCOUNT_NAME
STORAGE_ACCOUNT_KEY

Based upon:
https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python
"""
import os
from azure.storage.blob import BlockBlobService
import argparse

def arg_parse():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='This script is for downloading blob files from a blob storage container on Azure.')
    parser.add_argument("--container", dest='container', help="Blob storage container name", type=str)
    parser.add_argument("--output-dir", dest='output_dir', help="Local folder to which files will be saved", type=str)
    return parser.parse_args()

args = arg_parse()
os.makedirs(args.output_dir, exist_ok=True)

block_blob_service = BlockBlobService(account_name=os.getenv('STORAGE_ACCOUNT_NAME'), 
                                      account_key=os.getenv('STORAGE_ACCOUNT_KEY'))

container_name = args.container

generator = block_blob_service.list_blobs(container_name)
for blob in generator:
    print("\t Blob name: " + blob.name)
    os.makedirs(os.path.join(args.output_dir, os.path.dirname(blob.name)), exist_ok=True)
    # Download the blob(s).
    block_blob_service.get_blob_to_path(container_name, blob.name, os.path.join(args.output_dir, blob.name))