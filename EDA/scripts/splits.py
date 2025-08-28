import os
import argparse
import csv
from collections import defaultdict
from datetime import datetime

class FileData:
    def __init__(self, year, month, day, time, bbox_number, path):
        self.year = year
        self.month = month
        self.day = day
        self.time = time
        self.bbox_number = bbox_number
        self.path = path

def parse_filename(filename, root):
    """
    Parses a filename like 'sevemos_2008-06-15_05:55:00_2.npy' into its components.
    """
    if not filename.endswith(".npy"):
        return None

    try:
        parts = filename.split("_")
        date_str = parts[1]
        time_str = parts[2]
        bbox = parts[3].split(".")[0]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return FileData(
            year=date.year,
            month=date.month,
            day=date.day,
            time=time_str,
            bbox_number=bbox,
            path=os.path.join(root, filename)
        )
    except Exception as e:
        print(f"❌ Could not parse {filename}: {e}")
        return None

def group_files_by_day(data_directory, exclude_years=set()):
    """
    Groups .npy files into a dict[day_of_month] = [FileData, ...]
    """
    files_by_day = defaultdict(list)

    for root, _, files in os.walk(data_directory):
        for fname in files:
            f = parse_filename(fname, root)
            if f and f.year not in exclude_years:
                files_by_day[f.day].append(f)

    return files_by_day

def save_split_as_csv(split_name, files, output_csv_filename):
    with open(output_csv_filename, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["split", "year", "month", "day", "bbox_number", "path"])

        for f in files:
            rel_path = os.path.join(str(f.year), os.path.basename(f.path))
            writer.writerow([split_name, f.year, f.month, f.day, f.bbox_number, rel_path])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="Projects/datasets/third_dataset/csv")
    parser.add_argument("--val_days", nargs='+', type=int, default=[2, 7, 14, 18, 25, 30])
    parser.add_argument("--exclude_year", type=int, default=2019)

    args = parser.parse_args()
    print(f"Current working directory: {os.getcwd()}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("📂 Grouping files by day...")
    files_by_day = group_files_by_day(args.data_dir, exclude_years={args.exclude_year})

    train_files, val_files = [], []

    for day, files in files_by_day.items():
        if day in args.val_days:
            val_files.extend(files)
        else:
            train_files.extend(files)

    print(f"✅ Train files: {len(train_files)}")
    print(f"✅ Val files:   {len(val_files)}")

    save_split_as_csv("train", train_files, os.path.join(args.output_dir, "train_split.csv"))
    save_split_as_csv("val", val_files, os.path.join(args.output_dir, "val_split.csv"))
    print("✅ Splits saved successfully!")

def test_split():
    base_dir = "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/fused_dataset/dataset/2019"
    files = []

    for fname in os.listdir(base_dir):
        if fname.endswith(".npy"):  # Probably meant .npy, not .csv
            fdata = parse_filename(fname, base_dir)
            if fdata:
                files.append(fdata)

    save_split_as_csv(
        "test",
        files,
        "/net/nfs/ssd3/cfrancoismartin/Projects/datasets/fused_dataset/csv/test_split.csv"
    )
if __name__ == "__main__":
    main()
    test_split()
