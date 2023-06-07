'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import subprocess
import profile_utils as prof
import ci_utils as ci

"""
USER CONSTRAINTS
"""
timeout = 300


"""
BACKEND
"""
ci_cwd = os.getcwd()
test_cwd = os.getcwd()
trainlib_cwd = os.getcwd() + "/../../lib"
results_file = ci_cwd + "/test_suite_results.txt"
checkpoint = ci_cwd + "/checkpoint.txt"

with open(results_file, 'w') as f:

    print("<<< ENTERING TEST SEQUENCE FOR CONTINUOUS INTEGRATION >>>")

    # Create the temp folder
    if not os.path.exists(ci_cwd+"/temp"):
        os.mkdir(ci_cwd+"/temp")   
    if not os.path.exists(ci_cwd+"/temp/tests"):
        os.mkdir(ci_cwd+"/temp/tests")
    if not os.path.exists(ci_cwd+"/temp/lib"):
        os.mkdir(ci_cwd+"/temp/lib")
    
    # Go to the test folder
    os.chdir(ci_cwd+"/../../tests/")
    test_cwd = os.getcwd()

    print("CI Suite Folder: "+ci_cwd)
    print("Test Folder: "+test_cwd)
    print("TrainLib Folder: "+trainlib_cwd)

    # Copy PULP-TrainLib in the right position
    ci.copy_trainlib_ci(ci_cwd, trainlib_cwd)

    """
    START TEST SEQUENCE
    """
    test_sequence_iterator = 0

    print("\n=====> ENTERING TEST SEQUENCE FOR IM2COL.. <=====\n")

    current_test_source_folder = test_cwd + "/test_im2col"
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    os.system("rm -r BUILD/")
    cmd = "make clean all run > log.txt"
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\nim2col check...\n", 0, results_file)
    test_sequence_iterator += 1

