# data_folder_path = f"/Users/cp68wp/Documents/GitHub/Biogeography/data"
import os

match = "EcoGP"
data_folder_path = os.path.join("/", os.getcwd().split(match)[0], match, "data")
