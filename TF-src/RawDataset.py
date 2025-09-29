# KDDTest.txt [Cloumn , Rows : 11851 , 45 ]
# https://www.kaggle.com/datasets/primus11/nsl-kdd-dataset-filtered-version-of-kdd

column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv",
    "label", "difficulty"]

# Import the pandas library and give it the alias 'pd'
# Pandas is used for data manipulation and analysis, especially with tabular data like CSV or Excel files
# pandas is essential for working with datasets in machine learning and cybersecurity analysis

import pandas as pd
# Load the dataset
df = pd.read_csv("KDDTest.txt",names=column_names,index_col=False)
# Set max_columns to a higher value (e.g., 50 or None for unlimited)
pd.set_option('display.max_columns', 45)  
# Display first 10 rows
df.head(5)

# Export first 5 rows to Excel
df.head(5).to_excel("KDDTest_preview.xlsx", index=False)

#  Export All KDDTest.txt to Excel with Rows 11851 
df.to_excel("KDDTest_full.xlsx", sheet_name="Threats", index=False)

df = pd.read_csv("KDDTest.txt", names=column_names)
# Reset index if needed
df.reset_index(drop=True, inplace=True)

# Now Export All KDDTest.txt to Excel with Preview-10 rows & Full-11851 rows  
with pd.ExcelWriter("KDDTest_multi.xlsx") as writer:
    df.head(10).to_excel(writer, sheet_name="Preview", index=False)
    df.to_excel(writer, sheet_name="Full", index=False)
  
# Import the built-in 'os' module, which provides functions for interacting with the operating system
# Print the current working directory , this shows the folder where your Python script or notebook is running
import os
print(os.getcwd())



