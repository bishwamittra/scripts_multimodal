import os


def tar_files_and_folders(tar_name, file_list):
    cmd = 'tar -czvf ' + tar_name + '.tar.gz'
    for file in file_list:
        cmd += ' ' + file
    os.system(cmd)


tar_files_and_folders('send_to_git', 
                    ['*.txt', 
                     '*.py',
                     '*.sh',
                     '*.ipynb',
                     'classification',
                     'rescalenet',
                     'configs'
                     ])




# 4.0K    ReadMe.txt
# 136K    __pycache__
# 24K     _compile_u_shaped.py
# 4.0K    _todo_compile.sh
# 341M    cifar10_data
# 380K    classification
# 24K     client_split.ipynb
# 32K     client_split_u_shaped.ipynb
# 24K     compile.py
# 24K     compile.txt
# 24K     compile2.py
# 24K     compile_u_shaped.py
# 2.1M    configs
# 12K     rescale.py
# 1.3G    rescale18_split_test
# 1.3G    rescale18_split_u_shaped
# 48K     rescale_estimator.py
# 48K     rescale_estimator_u_shaped.py
# 12K     rescale_u_shaped.py
# 52K     rescalenet
# 16K     server_split.py
# 4.0K    test.py
# 4.0K    todo_run.sh
# 4.0K    utils.py