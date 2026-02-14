import os
import json
import sys
from tqdm import tqdm

def collect_depth3_names(base_folder: str) -> list[str]:
    base_folder = os.path.abspath(base_folder)
    base_depth = base_folder.rstrip(os.sep).count(os.sep)

    names = []
    for root, dirs, _files in tqdm(os.walk(base_folder), desc="Collecting sequence names"):
        # depth relative to base_folder
        rel_depth = root.rstrip(os.sep).count(os.sep) - base_depth

        # We only care about the parents of depth-3 directories (i.e., depth == 2)
        # because their children dirs are exactly depth 3.
        if rel_depth == 2:
            for d in dirs:
                names.append(d)
                if len(names) % 100 == 0:
                    print(f"Collected {len(names)} sequence names so far...")
                    print(f"Current sequence name: {d}")
            # Optional: don't descend further once we hit depth 2
            # (we already grabbed depth-3 names from `dirs`)
            dirs[:] = []

    return names


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <base_folder_path> <output_json_path>")
        sys.exit(1)

    base_folder = sys.argv[1]
    output_json = sys.argv[2]

    if not os.path.isdir(base_folder):
        print(f"Error: '{base_folder}' is not a valid directory.")
        sys.exit(1)

    seq_names = collect_depth3_names(base_folder)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(seq_names, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(seq_names)} sequence names (depth 3) to {output_json}")


if __name__ == "__main__":
    main()
