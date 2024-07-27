from tools.filetools import FileTools
from tools.datautils import DataUtils
from os.path import join, split, exists
import os , json
import numpy as np

GROUP = "Mindfulness-Project"

ftools = FileTools(GROUP)
dutils = DataUtils()
ROOTFOLDER = join(dutils.DATAPATH,GROUP)
QUALITY_FILE = join(dutils.DEVANALYSEPATH,"data","quality",f"{GROUP}.txt")


def get_desc_value(session):
    return {session:{"dwi":0,"anat":0,"spectroscopy":0,"fmap":0}}


def list_subject_sessions_with_spectroscopy_zero(description):
    """
    Lists all subject IDs and their sessions where spectroscopy is 0.

    Parameters:
    - description (dict): A dictionary with subject IDs as keys and session data as values.

    Returns:
    - dict: A dictionary where keys are subject IDs and values are lists of sessions with spectroscopy equal to 0.
    """
    result = {}
    for subject_id, sessions in description.items():
        zero_spectroscopy_sessions = [session for session, data in sessions.items() if data.get('spectroscopy') == 0]
        if zero_spectroscopy_sessions:
            result[subject_id] = zero_spectroscopy_sessions
    return result

def order_dict_by_keys(description):
    """
    Orders the dictionary by subject ID keys in ascending order starting with the lowest number.

    Parameters:
    - description (dict): A dictionary with subject IDs as keys and session data as values.

    Returns:
    - dict: A dictionary ordered by subject IDs.
    """
    # Sort the keys based on the numeric part of the subject IDs
    sorted_keys = sorted(description.keys(), key=lambda x: int(x[1:]))
    # Create a new dictionary ordered by the sorted keys
    ordered_description = {key: description[key] for key in sorted_keys}
    return ordered_description


def process_table_file_to_list(file_path):
    """
    Reads a table from a file and creates a list of dictionaries with keys: ID, Session, Value.
    """
    # Initialize an empty list to store the results
    result = []
    subject_id_arr = list()
    session_arr = list()
    quality_arr = list()
    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Process each line
        for line in lines:
            # Strip any leading/trailing whitespace and split the line into components
            parts = line.strip().split("\t")
            # Create a dictionary for each line
            subject_id = f"S{parts[0][4::]}".replace(" ","")
            session = int(parts[1])+1
            quality = parts[2]
            if parts[2]=="NA":
                quality = 0
            elif parts[2]=="0":
                quality = -1
            elif parts[2]=="0.5":
                quality = 0.5
            elif parts[2]=="1":
                quality = 1
            subject_id_arr.append(subject_id)
            session_arr.append(f"V{session}")
            quality_arr.append(quality)
    return np.array(subject_id_arr),np.array(session_arr),np.array(quality_arr)


q_subject_id_arr,q_session_arr,quality_arr = process_table_file_to_list(QUALITY_FILE)

description = dict()
for subject_id_el in os.listdir(ROOTFOLDER):
    if "sub-" in subject_id_el:
        subject_id = subject_id_el[4::]
        # if subject_id!="S069":continue
        desc_value=dict()
        for session_el in os.listdir(join(ROOTFOLDER,subject_id_el)):
            if "ses-" in session_el:
                session = session_el[4::]
                _desc_value = get_desc_value(session)
                for mri in _desc_value[session].keys():
                    if exists(join(ROOTFOLDER,subject_id_el,session_el,mri)):
                        _desc_value[session][mri]=1
                    else:
                        _desc_value[session][mri]=0
                    if mri=="spectroscopy":
                        indices = np.where((q_subject_id_arr==subject_id) & (q_session_arr==session))[0]
                        if len(indices)!=0:
                            indices = indices[0]
                            _desc_value[session][mri]=quality_arr[indices]
                desc_value.update(_desc_value)
        description[subject_id] = desc_value


description  = order_dict_by_keys(description)
outpath  = join(dutils.DATAPATH,GROUP,"mrsi_quality_check.json")

with open(outpath, 'w') as json_file:
    json.dump(description, json_file, indent=4)  # indent=4 for pretty printing
     

