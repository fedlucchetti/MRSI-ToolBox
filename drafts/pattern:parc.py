import re

def extract_substring(input_string):
    # Define regex patterns for the two types of substrings
    pattern1 = r'geometric_cubeK23mm'
    pattern2 = r'chimeraLFMIHIFIF-\d+'
    
    # Search for the patterns in the input string
    match1 = re.search(pattern1, input_string)
    match2 = re.search(pattern2, input_string)
    
    # Return the matched substring
    if match1:
        return match1.group()
    elif match2:
        return match2.group()
    else:
        return None

# Test examples
test_strings = [
    "sub-S038_ses-V1_run-01_acq-memprage_atlas-geometric_cubeK23mm_dseg_simmatrix",
    "sub-S038_ses-V1_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale3grow2mm_dseg_simmatrix",
    "sub-S038_ses-V1_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale1grow2mm_dseg_simmatrix",
    "sub-S038_ses-V1_run-01_acq-memprage_atlas-chimeraLFMIHIFIF_desc-scale5grow2mm_dseg_simmatrix"
]

# Run the function and print results
for test_str in test_strings:
    print(extract_substring(test_str))