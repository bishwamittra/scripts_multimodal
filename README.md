This is a repository for split learning of multi-modal medical image classification. 

## Installations
`pip install requirements.txt`


## Data
Download Seven point checklist criteria dataset from this [link](https://derm.cs.sfu.ca/Welcome.html) and unzip it in a directory `data/`. In fact, there should a directory `data/release_v0` for the code to work.


## Reproducibility
Run scripts inside `fusionnet_split_learning` directory.
- Run non-split model `python main.py`.
- Before running the split model, make sure to properly set host and port of the network. Look inside `client_u_shaped.py` and `server_u_shaped.py`.
    - Start with server `python server_u_shaped.py`.
    - Then run client `python client_u_shaped.py`.


