"""
Check the logs of a cloud deployed Azure ML webservice in a workspace
"""
from azureml.core import Workspace
from azureml.core.webservice import Webservice


ws = Workspace.from_config(path='./.azureml/config.json')
webservices = ws.webservices

for name, webservice in webservices.items():
    print("Webservice: {}, scoring URI: {}".format(name, webservice.scoring_uri))
    if name == "keras-yolov3-firearms":
        print(webservice.get_logs())
        print(list(webservice.get_keys()))
        print(webservice.scoring_uri)
