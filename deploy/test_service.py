"""
Use Python requests to call web service
"""
import requests
from requests.exceptions import HTTPError
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from PIL import Image
from io import BytesIO
import base64


def arg_parse():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(
        description='This script is for calling the HTTP REST API for Azure ML cloud service')
    parser.add_argument("--image", dest='image', 
        help="Image file", type=str)
    return parser.parse_args()

def get_service(ws, name):
    """Get the web service object from Azure ML workspace that matches name"""
    webservices = ws.webservices
    service = None
    for servicename, webservice in webservices.items():
        if name == servicename:
            service = webservice
            break
    return service

def main(img_file):
    """Call Azure ML webservice, sending image and returning inference results"""

    scoring_uri = os.getenv('SCORING_URI')

    # Construction input data json string
    # input_data = plt.imread(img_file)
    # input_data = {"data": [input_data.tolist()] }

    pil_img = Image.open(img_file)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = json.dumps({'data': base64.b64encode(buff.getvalue()).decode("utf-8")})
    with open('request.txt', 'w') as f:
        f.write(new_image_string)

    headers = {'Content-Type':'application/json'}#, 'Authorization': 'Bearer ' + os.getenv('WEBSERVICE_KEY')}

    try:

        resp = requests.post(scoring_uri, 
                             new_image_string, 
                             headers=headers)

        # If the response was successful, no Exception will be raised
        resp.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('Success!')

if __name__ == "__main__":
    args = arg_parse()
    main(args.image)
    