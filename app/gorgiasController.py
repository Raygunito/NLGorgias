import json
import os
import requests  # Install requests using 'pip install requests'
from io import StringIO
from dotenv import load_dotenv

# par défaut c'est ".env" mais on peut le changer
# load_dotenv(dotenv_path="./.env.local", override=True)
load_dotenv()

BASE_GORGIAS_URL = "http://aiasvm1.amcl.tuc.gr:8085"

username = os.getenv("USER")
password = os.getenv("PASSWORD")


class GorgiasController:
    def __init__(self, project_name="gorgias_project"):
        self.project_name = project_name
        self.createProject(project_name)
        self.deleted = False

    def sendContent(self, content, file_name="gorgias_file", type="gorgias"):
        """
        Send content to Gorgias server as a file.
        :param content: The content to be sent (string).
        :param file_name: The name of the file to be created.
        :param type: The type of the file (e.g., "gorgias", "prolog").
        :return: The response from the server.
        """
        files = {'file': (file_name, StringIO(content), 'text/plain')}

        self.file_name = file_name

        r = requests.post(f"{BASE_GORGIAS_URL}/addFile?project={self.project_name}&type={type}",
                          files=files, auth=(username, password))
        if r.status_code != 200:
            print(r.status_code)
            return False
        return r.json()

    def query(self, facts=[], query=""):
        if not self.file_name:
            print("content not sent before")
            return False
        fileToUse = self.project_name+"/"+self.file_name
        data = {
            "facts": facts,
            "gorgiasFiles": [fileToUse],
            "query": query,
            "resultSize": 1
        }

        r = requests.post(f"{BASE_GORGIAS_URL}/GorgiasQuery",
                          json=data, auth=(username, password))

        if r.status_code != 200:
            print("error during query")
            return False
        return r.json()

    def createProject(self, project_name="default_project"):
        r = requests.post(
            f"{BASE_GORGIAS_URL}/createProject?project_name={project_name}", auth=(username, password))
        if r.status_code != 200:
            print("error while creating project")
            return False
        self.project_name = project_name
        return r.json()

    def deleteProject(self):
        r = requests.post(
            f"{BASE_GORGIAS_URL}/deleteProject?project={self.project_name}", auth=(username, password))
        if r.status_code != 200:
            print("error couldn't delete project")
            return False
        self.deleted = True
        return r.json()


if __name__ == "__main__":
    exampleContent = """
    :- dynamic phone_call/0, at_work/0, family_member/0, at_meeting/0.
    rule(r1, allow, []):- phone_call.
    rule(r2, deny, []):- phone_call.
    rule(p1, prefer(r1, r2), []).
    rule(p2, prefer(r2, r1), []):- at_work.
    rule(c1, prefer(p2, p1), []).
    rule(c2, prefer(p1, p2), []):- family_member.
    rule(c3, prefer(c2, c1), []).
    rule(c4, prefer(c1, c2), []):- at_meeting.
    rule(c5, prefer(c4, c3), []).
    complement(deny, allow).
    complement(allow, deny)."""

    gorgias = GorgiasController("monSuperProjet")
    gorgias.sendContent(exampleContent, "fichier_tmp", "gorgias")

    response = gorgias.query(
        facts=["phone_call", "at_work", "family_member"], query="allow")
    print(json.dumps(response, indent=2))
    gorgias.deleteProject()


# unused functions
# Exemple of a function using the prolog API to add a file
def addFile(file="", project="", type=""):
    files = {'file': open(f'{file}', 'rb')}

    r = requests.post(f"{BASE_GORGIAS_URL}/addFile?project={project}&type={type}",
                      files=files, auth=(username, password))
    if r.status_code != 200:
        print(r.status_code)
        return False
    return r.json()

# Exemple of a function using the prolog API to delete a file


def deleteFile(filename="", project="", ):
    r = requests.post(
        f"{BASE_GORGIAS_URL}/deleteFile?filename={filename}.pl&project={project}", auth=(username, password))
    if r.status_code != 200:
        print("error")
        return False
    return r.json()
