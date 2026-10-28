# Imports

# 1. Standard library
import os
import shutil

# 2. Third-party
# None


# Functions

def gather_patient_data(sampletypes, results_path):
    """
    Gather patient sample data from Plasma and Urine folders into a unified patient directory.

    For each unique patient ID, creates a subfolder under results_path/Patients/ and copies
    all corresponding sample folders from Plasma and Urine. Deletes the original Plasma and
    Urine directories afterwards.

    Parameters:
    -----------
    sampletypes : list of str
        Sample types to include; 'P' for plasma, 'U' for urine
    results_path : str
        Root directory containing Plasma and Urine subdirectories

    Returns:
    --------
    None
    """
    print('Gather sample data in patient folders.')
    # Define patient and sample info paths
    out_path = results_path + '/Patients/'
    
    ### Check if output directory already exists
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    ### list directories in Plasma and Urine folders
    p_dir = results_path + '/Plasma/Patients/'
    u_dir = results_path + '/Urine/Patients/'
    if os.path.isdir(p_dir):
        p_samples = os.listdir(p_dir)
    else:
        p_samples = []
        print('Error: No Plasma data in folder.')
    if os.path.isdir(u_dir):
        u_samples = os.listdir(u_dir)
    else:
        u_samples = []
        print('Error: No Urine data in folder.')
    
    ### Gather all patient IDs
    all_samples = p_samples + u_samples
    
    # Extract unique patient IDs to avoid duplicates
    unique_patient_ids = set()
    for sample in all_samples:
        if len(sample) >= 5:  # Ensure sample name is long enough
            patient_id = sample[2:5]
            unique_patient_ids.add(patient_id)
    
    ### For each unique patient, gather all sample folders
    for patient_id in sorted(unique_patient_ids):
        if not os.path.isdir(out_path + patient_id):
            os.makedirs(out_path + patient_id)
        
        # Find all samples for this specific patient
        patient_samples = [x for x in all_samples if len(x) >= 5 and x[2:5] == patient_id]
        
        for sample in patient_samples:
            # gather P samples and copy to new Patient folder
            if 'P' in sampletypes:
                if sample in p_samples:
                    path = results_path + '/Plasma/Patients/' + sample
                    target_path = out_path + patient_id + '/' + sample
                    if not os.path.isdir(target_path):
                        shutil.copytree(path, target_path)
            
            # gather U samples and copy to new Patient folder
            if 'U' in sampletypes:
                if sample in u_samples:
                    path = results_path + '/Urine/Patients/' + sample
                    target_path = out_path + patient_id + '/' + sample
                    if not os.path.isdir(target_path):
                        shutil.copytree(path, target_path)
    
    ### Delete Plasma and Urine directories
    shutil.rmtree(results_path + '/Urine')
    shutil.rmtree(results_path + '/Plasma')


