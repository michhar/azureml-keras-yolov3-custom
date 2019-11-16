import os
from azure.storage.blob import BlockBlobService, PublicAccess
import argparse
import glob

def arg_parse():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='This script is for uploading a directory to Azure Blob Storage.')
    parser.add_argument("--container", dest='container', help="Blob storage container name", type=str)
    parser.add_argument("--account", dest='account', help="Storage account name", type=str)
    parser.add_argument("--key", dest='key', help="Storage account key", type=str)
    parser.add_argument("--dir", dest='directory', help="The directory to upload")
    return parser.parse_args()

args = arg_parse()

# Create the BlockBlockService that is used to call the Blob service for the storage account
block_blob_service = BlockBlobService(account_name=args.account, account_key=args.key) 

# Create a container
container_name = args.container
block_blob_service.create_container(container_name) 

# Set the permission so the blobs are public.
block_blob_service.set_container_acl(container_name, public_access=PublicAccess.Container)

for filename in glob.iglob(os.path.join(args.directory, '**', '*'), recursive=True):
    if os.path.isfile(filename):
        print('Uploading ', filename)
        # Upload the created file, use local_file_name for the blob name
        block_blob_service.create_blob_from_path(container_name, filename, filename)

# Check that the files uploaded correctly to blob
generator = block_blob_service.list_blobs(container_name)
for blob in generator:
    print("Blob name in Azure: " + blob.name)