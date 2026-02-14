import pandas as pd, sqlite3, os
from uco3d.dataset_utils.utils import get_dataset_root
import json
UCO3D_DATASET_ROOT = get_dataset_root(assert_exists=True)
# read the main metadata table (takes long time)
metadata_file = os.path.join(UCO3D_DATASET_ROOT, "metadata.sqlite")
frame_annots = pd.read_sql_table("frame_annots", f"sqlite:///{metadata_file}")

available_seq_json_path = "/home/stud/kandelak/git/uco3d/available_sequences.json"
available_seqs = list(json.load(open(available_seq_json_path, "r"))) # simply loads list of strings


available_setlist = frame_annots[frame_annots["sequence_name"].isin(available_seqs)][["frame_number","sequence_name"]]
available_setlist["subset"] = "all"


out_dir = "./"

setlist_dir = os.path.join(out_dir, "set_lists")
os.makedirs(setlist_dir, exist_ok=True)  # <-- create directory if needed

setlist_file = os.path.join(setlist_dir, "set_lists_available_on_server.sqlite")
# store the new table
with sqlite3.connect(setlist_file) as con:
    available_setlist.to_sql("set_lists", con, if_exists='replace', index=False)