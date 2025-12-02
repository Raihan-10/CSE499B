# Reproducibility
random.seed(42) #for keeping random values same

# input dir
DATASET_ROOT = "/kaggle/input/smart-traffic"

# Working dirs
MERGED_IMAGES_DIR = "Unified_Images"
MERGED_ANNOTS_DIR = "Unified_Annotations"
STD_ANNOTS_DIR    = "Standardized_Annotations"
FINAL_OUTPUT_DIR  = "Final_YOLO_Dataset"

IMAGE_EXTENSIONS    = (".jpg", ".jpeg", ".png")
ANNOTATION_EXTS_RAW = (".txt", ".xml") 

os.makedirs(MERGED_IMAGES_DIR, exist_ok=True)
os.makedirs(MERGED_ANNOTS_DIR, exist_ok=True)

print("DATASET_ROOT:", DATASET_ROOT)
print("Working dir:", os.getcwd()) #get current working dir

# scan data.yaml files, collect original classes

yaml_paths = glob.glob(os.path.join(DATASET_ROOT, "**", "data.yaml"), recursive=True) #** for finding all sub folders recursive true means it will search also nested loop
print(f"Found {len(yaml_paths)} data.yaml files")

ALL_ORIGINAL_NAMES = set()
YAML_PREFIX_TO_NAMES = {} 

for ypath in yaml_paths:
    try:
        with open(ypath, "r") as f:
            cfg = yaml.safe_load(f) #full dic of yaml file
        names = cfg.get("names", [])
        if isinstance(names, dict):
            names = list(names.values())
        names = [str(n) for n in names]

        # derive prefix similar to how it will merge
        path_of_yaml = os.path.dirname(ypath)
        parts = path_of_yaml.split(os.sep)
        relevant_parts = [
            p for p in parts
            if p.lower() not in ["images", "labels", "annotations", "train", "test", "valid"] and p
        ]
        prefix = relevant_parts[-1] if relevant_parts else "source"

        YAML_PREFIX_TO_NAMES[prefix] = names
        for n in names:
            ALL_ORIGINAL_NAMES.add(n)

        print(f"[YAML] {ypath}")
        print(f"  prefix: {prefix}")
        print(f"  names: {names}")
    except Exception as e:
        print(f"[ERROR] reading {ypath}: {e}")

print("\n=== UNIQUE ORIGINAL CLASS NAMES ACROSS ALL DATASETS ===")
for cls in sorted(ALL_ORIGINAL_NAMES):
    print(" -", cls)


#final class
FINAL_CLASS_NAMES = [
    "ambulance",           # 0
    "enforcement_vehicle", # 1
    "e_rickshaw",          # 2
    "bike_helmet",         # 3
    "no_helmet",           # 4
    "illegal_parking",     # 5
    "red_light",           # 6
    "green_light",         # 7
    "yellow_light",        # 8
    "stop_line",           # 9
    "car",                 # 10
    "bike",                # 11
    "bicycle",             # 12
    "bus",                 # 13
    "truck",               # 14
    "van",                 # 15
    "cng",                 # 16
    "rickshaw",            # 17
    "tractor",             # 18
    "horse_cart",          # 19
    "wheelbarrow",         # 20
    "leguna"               # 21
]

NAME_TO_ID = {name: i for i, name in enumerate(FINAL_CLASS_NAMES)}



# alias mapping
ALIASES = {
    # from vehicle folder
    "horsecart": "horse_cart",      
    "easybike": "e_rickshaw",       
    "easy-bike": "e_rickshaw",      
    "leguna": "leguna",
    "rickshaw": "rickshaw",
    "tractor": "tractor",
    "wheelbarrow": "wheelbarrow",
    "bicycle": "bicycle",
    "bike": "bike",
    "bus": "bus",
    "car": "car",
    "cng": "cng",
    "truck": "truck",
    "van": "van",
    
    
    "Horse-cart": "horse_cart",
    "Easy-bike": "e_rickshaw",
    "Leguna": "leguna",
    "Wheelbarrow": "wheelbarrow",
    "Bicycle": "bicycle",
    "Bike": "bike",
    "Bus": "bus",
    "Car": "car",
    "Cng": "cng",
    "Rickshaw": "rickshaw",
    "Tractor": "tractor",
    "Truck": "truck",
    "Van": "van",

    #  from yaml files
    "APC": "enforcement_vehicle",
    "Gun-Artillery": "enforcement_vehicle",
    "Rocket-Artillery": "enforcement_vehicle",
    "Military-Car": "enforcement_vehicle",
    "Military-Truck": "enforcement_vehicle",
    "Tank": "enforcement_vehicle",
    "PoliceCar": "enforcement_vehicle",
    "police_car": "enforcement_vehicle", 
    "-": "enforcement_vehicle",
    "8-28 train - v1 2024-08-28 5-13pm": "enforcement_vehicle",
    "This dataset was exported via roboflow.com on August 28- 2024 at 5-14 PM GMT": "enforcement_vehicle",

    "E-Rickshaw": "e_rickshaw",
    "Ambulance": "ambulance",
    "illegal parking": "illegal_parking",
    "illegalParking": "illegal_parking",
    "With Helmet": "bike_helmet",
    "Without Helmet": "no_helmet",

    "green_light": "green_light",
    "red_light": "red_light",
    "yellow_light": "yellow_light",
    "stop_line": "stop_line",
    "motobike": "bike",
}

#  logic
def original_name_to_final_id(name: str):
    """
    Maps class strings to ID with high performance.
    """
    if name is None: return None
    
    key = str(name).strip()

    # 1. Exact Alias Match (Priority)
    if key in ALIASES:
        target = ALIASES[key]
        return NAME_TO_ID.get(target)

    # 2. Exact Final ID Match
    if key in NAME_TO_ID:
        return NAME_TO_ID[key]
        
    # 3. Lowercase Fallback (Safety)
    key_lower = key.lower()
    
    if key_lower in ALIASES:
        target = ALIASES[key_lower]
        return NAME_TO_ID.get(target)
        
    if key_lower in NAME_TO_ID:
        return NAME_TO_ID[key_lower]

    return None

# verifying
if 'ALL_ORIGINAL_NAMES' in locals():
    unmapped = []
    print("--- Verification Results ---")
    for n in sorted(ALL_ORIGINAL_NAMES):
        mid = original_name_to_final_id(n)
        if mid is None:
            unmapped.append(n)
            print(f" {n} -> UNMAPPED")
        else:
            pass

    if unmapped:
        print(f"\n Found {len(unmapped)} unmapped classes! Update ALIASES.")
    else:
        print("\n All classes mapped successfully")
else:
    print("ALL_ORIGINAL_NAMES variable not found.")

print(f"{'ORIGINAL NAME (In your files)':<50} | {'FINAL YOLO CLASS'}")
print("-" * 80)

# specific check for the "Vehicle" folder
vehicle_folders = [
    "horsecart", "easybike", "leguna", "rickshaw", "tractor", 
    "wheelbarrow", "bicycle", "bike", "bus", "car", "cng", "truck", "van",
    "Horse-cart", "Easy-bike", "Leguna", "Wheelbarrow", "Bicycle", "Bike", 
    "Bus", "Car", "Cng", "Rickshaw", "Tractor", "Truck", "Van"
]

# Combine the ones found in your scan with the folder names
check_list = sorted(list(set(list(ALL_ORIGINAL_NAMES) + vehicle_folders)))

for name in check_list:
    fid = original_name_to_final_id(name)
    if fid is not None:
        final_name = FINAL_CLASS_NAMES[fid]
        print(f"{name:<50} | {final_name} ({fid})")
    else:
        print(f"{name:<50} |  UNMAPPED (Will be skipped)")

VEHICLE_FOLDER = "/kaggle/input/smart-traffic/Vehicle" 
all_names = set()

# Scan folder names
for name in os.listdir(VEHICLE_FOLDER):
    if os.path.isdir(os.path.join(VEHICLE_FOLDER, name)):
        all_names.add(name)

# Check mapping
for n in sorted(all_names):
    mapped_id = original_name_to_final_id(n)
    target_name = FINAL_CLASS_NAMES[mapped_id] if mapped_id is not None else "None"
    print(f"{n} -> {target_name} ({mapped_id})")
# merge and rename all images + raw annotations


# These should already be defined; redefine safely
MERGED_IMAGES_DIR = "Unified_Images"
MERGED_ANNOTS_DIR = "Unified_Annotations"

IMAGE_EXTENSIONS    = (".jpg", ".jpeg", ".png", ".webp")
ANNOTATION_EXTS_RAW = (".txt", ".xml")

# Clean and recreate merged directories
for d in [MERGED_IMAGES_DIR, MERGED_ANNOTS_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

for root, _, files in os.walk(DATASET_ROOT):
    parts = root.split(os.sep)

    # remove generic subdirs from prefix
    relevant_parts = [
        p for p in parts
        if p.lower() not in ["images", "labels", "annotations", "train", "test", "valid"] and p
    ]
    prefix = relevant_parts[-1] if relevant_parts else "source"

    for filename in files:
        src_path = os.path.join(root, filename)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        new_filename = f"{prefix}_{name}{ext}"

        try:
            if ext in IMAGE_EXTENSIONS:
                shutil.copy2(src_path, os.path.join(MERGED_IMAGES_DIR, new_filename))
            elif ext in ANNOTATION_EXTS_RAW:
                shutil.copy2(src_path, os.path.join(MERGED_ANNOTS_DIR, new_filename))
        except Exception as e:
            print(f"[COPY ERROR] {src_path}: {e}")

print("Merged images:", len(os.listdir(MERGED_IMAGES_DIR)))
print("Merged annotations:", len(os.listdir(MERGED_ANNOTS_DIR)))

# build PREFIX_TO_ORIGINAL_MAP only from real data.yaml files


PREFIX_TO_ORIGINAL_MAP = {}

for ypath in yaml_paths:
    try:
        with open(ypath, "r") as f:
            cfg = yaml.safe_load(f)

        names = cfg.get("names", [])
        if isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            id_to_name = {i: str(n) for i, n in enumerate(names)}
        else:
            continue

        path_of_yaml = os.path.dirname(ypath)
        parts = path_of_yaml.split(os.sep)
        relevant_parts = [
            p for p in parts
            if p.lower() not in ["images", "labels", "annotations", "train", "test", "valid"] and p
        ]
        prefix = relevant_parts[-1] if relevant_parts else "source"

        PREFIX_TO_ORIGINAL_MAP[prefix] = id_to_name
        print(f"[MAP] prefix '{prefix}' -> {id_to_name}")

    except Exception as e:
        print(f"[YAML MAP ERROR] {ypath}: {e}")

print("\nPREFIX_TO_ORIGINAL_MAP ready.")
# convert XML -> YOLO and standardize YOLO txt into unified ID space


STD_ANNOTS_DIR = "Standardized_Annotations"

if os.path.exists(STD_ANNOTS_DIR):
    shutil.rmtree(STD_ANNOTS_DIR)
os.makedirs(STD_ANNOTS_DIR, exist_ok=True)

def convert_xml_to_yolo(xml_path, out_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            cls_id = original_name_to_final_id(cls_name)
            if cls_id is None:
                continue

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            x_c = ((xmin + xmax) / 2.0) / img_w
            y_c = ((ymin + ymax) / 2.0) / img_h
            w   = (xmax - xmin) / img_w
            h   = (ymax - ymin) / img_h

            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        base = os.path.splitext(os.path.basename(xml_path))[0]
        out_txt = os.path.join(out_dir, f"{base}.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))

        return True
    except Exception as e:
        print(f"[XML ERROR] {xml_path}: {e}")
        return False


def standardize_yolo_txt(txt_path, out_dir):
    try:
        filename = os.path.basename(txt_path)

        # find best matching prefix (longest match at start followed by '_' or end)
        best_prefix = None
        for key in sorted(PREFIX_TO_ORIGINAL_MAP.keys(), key=len, reverse=True):
            if filename.startswith(key) and (len(filename) == len(key) or filename[len(key)] == "_"):
                best_prefix = key
                break

        if best_prefix is None:
            print(f"[PREFIX MISS] {filename}")
            return False

        src_map = PREFIX_TO_ORIGINAL_MAP[best_prefix]  # oldID -> originalName

        lines_out = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    old_id = int(parts[0])
                except ValueError:
                    continue

                orig_name = src_map.get(old_id)
                cls_id = original_name_to_final_id(orig_name)
                if cls_id is None:
                    continue

                parts[0] = str(cls_id)
                lines_out.append(" ".join(parts))

        if not lines_out:
            return False

        out_txt = os.path.join(out_dir, filename)
        with open(out_txt, "w") as f:
            f.write("\n".join(lines_out))

        return True
    except Exception as e:
        print(f"[TXT ERROR] {txt_path}: {e}")
        return False


xml_count, txt_count, skip_count = 0, 0, 0

for fname in os.listdir(MERGED_ANNOTS_DIR):
    path = os.path.join(MERGED_ANNOTS_DIR, fname)
    lower = fname.lower()

    if lower.endswith(".xml"):
        if convert_xml_to_yolo(path, STD_ANNOTS_DIR):
            xml_count += 1
        else:
            skip_count += 1
    elif lower.endswith(".txt") and not fname.startswith("README"):
        if standardize_yolo_txt(path, STD_ANNOTS_DIR):
            txt_count += 1
        else:
            skip_count += 1

print("XML converted:", xml_count)
print("YOLO txt standardized:", txt_count)
print("Skipped (errors / unmapped):", skip_count)

#repair and then again i have to run cell 6


MERGED_IMAGES_DIR = "Unified_Images"
MERGED_ANNOTS_DIR = "Unified_Annotations"

repaired_count = 0

print("--- Checking for Corrupt XMLs (Zero Width/Height) ---")

# Loop through all XMLs in the annotations folder
for filename in os.listdir(MERGED_ANNOTS_DIR):
    if not filename.lower().endswith('.xml'):
        continue
        
    xml_path = os.path.join(MERGED_ANNOTS_DIR, filename)
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        
        if size is not None:
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            
            # If dimensions are invalid (0), let's fix them
            if w <= 0 or h <= 0:
                # 1. Find corresponding image
                base_name = os.path.splitext(filename)[0]
                
                # Try different extensions to find the image
                found_img = None
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    img_path = os.path.join(MERGED_IMAGES_DIR, base_name + ext)
                    if os.path.exists(img_path):
                        found_img = img_path
                        break
                
                if found_img:
                    # 2. Read actual size from image
                    with Image.open(found_img) as img:
                        real_w, real_h = img.size
                    
                    # 3. Update XML
                    size.find("width").text = str(real_w)
                    size.find("height").text = str(real_h)
                    tree.write(xml_path)
                    
                    print(f"🔧 FIXED: {filename} (Updated size to {real_w}x{real_h})")
                    repaired_count += 1
                else:
                    print(f" Could not fix {filename}: Image not found.")

    except Exception as e:
        print(f"Error checking {filename}: {e}")

print(f"\nRepair Complete. Fixed {repaired_count} files.")

label_dir = "Standardized_Annotations"
removed = 0

for f in os.listdir(label_dir):
    path = os.path.join(label_dir, f)
    if os.path.isfile(path) and f.endswith(".txt"):
        with open(path, "r") as fp:
            content = fp.read().strip()
            if content == "":
                os.remove(path)
                removed += 1

print("Removed empty label files:", removed)

label_dir = "Standardized_Annotations"
empty_files = []

for f in os.listdir(label_dir):
    path = os.path.join(label_dir, f)
    if os.path.isfile(path) and f.endswith(".txt"):
        with open(path, "r") as fp:
            if fp.read().strip() == "":
                empty_files.append(f)

print("Empty label files:", len(empty_files))
for ef in empty_files:
    print(" -", ef)

# split into train/val/test and write final data.yaml



FINAL_OUTPUT_DIR = "Final_YOLO_Dataset"
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# Clean final dir
if os.path.exists(FINAL_OUTPUT_DIR):
    shutil.rmtree(FINAL_OUTPUT_DIR)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

splits = ["train", "val", "test"]
split_dirs = {
    s: {
        "images": os.path.join(FINAL_OUTPUT_DIR, s, "images"),
        "labels": os.path.join(FINAL_OUTPUT_DIR, s, "labels"),
    }
    for s in splits
}
for s in splits:
    os.makedirs(split_dirs[s]["images"], exist_ok=True)
    os.makedirs(split_dirs[s]["labels"], exist_ok=True)

# list merged images
all_imgs = [
    f for f in os.listdir(MERGED_IMAGES_DIR)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
]
all_basenames = [os.path.splitext(f)[0] for f in all_imgs]

# keep only images which have a standardized label
all_basenames = [
    b for b in all_basenames
    if os.path.exists(os.path.join(STD_ANNOTS_DIR, f"{b}.txt"))
]

random.shuffle(all_basenames)

train_b, temp_b = train_test_split(
    all_basenames,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=42,
)
val_b, test_b = train_test_split(
    temp_b,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=42,
)

split_map = {
    "train": train_b,
    "val": val_b,
    "test": test_b,
}

print("Total labeled images:", len(all_basenames))
print("Train:", len(train_b), "Val:", len(val_b), "Test:", len(test_b))

# map basename to original extension
ext_map = {
    os.path.splitext(f)[0]: os.path.splitext(f)[1]
    for f in all_imgs
}

for split, basenames in split_map.items():
    for b in basenames:
        img_ext = ext_map.get(b, ".jpg")

        src_img = os.path.join(MERGED_IMAGES_DIR, f"{b}{img_ext}")
        src_lbl = os.path.join(STD_ANNOTS_DIR, f"{b}.txt")

        dst_img = os.path.join(split_dirs[split]["images"], f"{b}{img_ext}")
        dst_lbl = os.path.join(split_dirs[split]["labels"], f"{b}.txt")

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# write final data.yaml
data_yaml = {
    "path": f"/kaggle/working/{FINAL_OUTPUT_DIR}",
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": len(FINAL_CLASS_NAMES),
    "names": FINAL_CLASS_NAMES,
}

yaml_path = os.path.join(FINAL_OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("Final dataset root:", FINAL_OUTPUT_DIR)
print("data.yaml written at:", yaml_path)
#chcking all classes total label


LABEL_DIR = "Standardized_Annotations"  # your YOLO txt folder
class_counts = Counter()

for fname in os.listdir(LABEL_DIR):
    if fname.endswith(".txt"):
        path = os.path.join(LABEL_DIR, fname)
        with open(path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1

# Print counts for all classes
for cls_id in range(len(FINAL_CLASS_NAMES)):
    print(f"Class {cls_id} ({FINAL_CLASS_NAMES[cls_id]}): {class_counts.get(cls_id, 0)} labels")

#checking how many hidden png

dataset_dir = "Final_YOLO_Dataset"

print(f" Deep Scanning {dataset_dir} (READ-ONLY MODE)...")

hidden_pngs = 0
iccp_warnings = 0
total_scanned = 0

for root, dirs, files in os.walk(dataset_dir):
    for file in tqdm(files, desc="Inspecting"):
        # Check all image types
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            total_scanned += 1
            img_path = os.path.join(root, file)
            
            try:
                with Image.open(img_path) as img:
                    # Check 1: Is it a PNG disguised as JPG?
                    # (File extension says .jpg, but internal format is PNG)
                    is_png_format = (img.format == 'PNG')
                    is_named_jpg  = file.lower().endswith(('.jpg', '.jpeg'))
                    
                    if is_png_format and is_named_jpg:
                        hidden_pngs += 1
                    
                    # Check 2: Does it have the bad profile?
                    if "iCCP" in img.info:
                        iccp_warnings += 1
                        
            except Exception as e:
                # print(f"Error reading {file}: {e}")
                pass

print("-" * 40)
print(f" Total images scanned: {total_scanned}")
print(f"  Hidden PNGs (JPG name, PNG content): {hidden_pngs}")
print(f"  Images with iCCP warnings: {iccp_warnings}")
print("-" * 40)

if hidden_pngs > 0 or iccp_warnings > 0:
    print(" Verdict: You HAVE issues. You should run the conversion script.")
else:
    print(" Verdict: Your data is clean. The warnings might be false alarms.")

#convert the png to jpg


dataset_dir = "Final_YOLO_Dataset"

print(f"🔧 Fixing 50 Hidden PNGs in {dataset_dir}...")

fixed_count = 0

for root, dirs, files in os.walk(dataset_dir):
    for file in tqdm(files, desc="Processing"):
        # Check only files that look like JPGs
        if file.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            
            try:
                img = Image.open(img_path)
                
                # IF it is actually a PNG inside...
                if img.format == 'PNG':
                    # 1. Convert to standard RGB (removes transparency/alpha)
                    img = img.convert("RGB")
                    
                    # 2. Save as a TRUE JPEG (strips all PNG metadata)
                    img.save(img_path, "JPEG", quality=95)
                    fixed_count += 1
                    
            except Exception as e:
                print(f"Error fixing {file}: {e}")

print("-" * 40)
print(f" Fixed {fixed_count} images.")
print("You can now restart training. The red warnings will be gone!")
#training
model = YOLO("yolov8m.pt")

model.train(
    data="Final_YOLO_Dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,       
    device=0,      
    workers=4,       
    patience=10,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    amp=True,
)
#evaluation using testing
!yolo detect val model=/kaggle/input/best-model1/other/best-model/1/best.pt data=/kaggle/working/fixed_data.yaml split=test

#performance check
# 1. Load the results file
results = pd.read_csv('/kaggle/input/performance/Training-result/results.csv')

# 2. Clean column names 
results.columns = [c.strip() for c in results.columns]

# 3. Create the Comparison Plot
plt.figure(figsize=(12, 5))

# Plot Box Loss 
plt.subplot(1, 2, 1)
plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss')
plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', linestyle='--')
plt.title('Box Loss (Location Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Class Loss
plt.subplot(1, 2, 2)
plt.plot(results['epoch'], results['train/cls_loss'], label='Train Cls Loss')
plt.plot(results['epoch'], results['val/cls_loss'], label='Val Cls Loss', linestyle='--')
plt.title('Class Loss (Identification Accuracy)')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#checking the map50 and map50-95 for testing

# 
model = YOLO('/kaggle/input/best-model1/other/best-model/1/best.pt')

#  Run the Test evaluation
metrics = model.val(data='/kaggle/working/fixed_data.yaml', split='test')

#  Extract the exact numbers
# metrics.box.map    = mAP50-95 (The strict metric)
# metrics.box.map50  = mAP50    (The loose metric)

print("\n" + "="*30)
print(f"YOUR TEST RESULTS:")
print(f"mAP@50-95: {metrics.box.map:.5f}") 
print(f"mAP@50:    {metrics.box.map50:.5f}")
print("="*30 + "\n")
#testing with image

# 1. Load the model

model = YOLO('/kaggle/input/best-model1/other/best-model/1/best.pt') 

# 2. Perform inference on an image source

source = '/kaggle/input/finaldataset/test/images/BikeHelmet_BikesHelmets100_png_jpg.rf.65bd2ec796d3529f827e480fe8bc6d1b.jpg' 
results = model(source)

# 3. Display and Save results
for result in results:
    # Show the image on your screen
    result.show() 
    
    # Save the resulting image with boxes drawn
    result.save(filename='prediction.jpg')
#testing with a video
# Configuration

model_path = '/kaggle/input/best-model1/other/best-model/1/best.pt'
video_path = '/kaggle/input/bikehelmet/bikeHelmet2.mp4'
temp_output = 'helmet.mp4'
final_output = 'final_output.mp4'

# Load YOLO Model
model = YOLO(model_path)

# Video Reader + Writer Setup
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    temp_output,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# YOLO Tracking 
print(f"Processing {total_frames} frames...")

results = model.track(
    source=video_path,
    persist=True,
    stream=True,
    imgsz=1280,      # best for helmet
    conf=0.25,
    iou=0.45,
    verbose=False
)

for result in tqdm(results, total=total_frames, unit="frame"):
    frame = result.plot()
    writer.write(frame)

writer.release()
cap.release()


# Convert to browser-compatible format
if os.path.exists(temp_output):
    os.system(
        f"ffmpeg -y -loglevel panic -i {temp_output} "
        f"-vcodec libx264 -pix_fmt yuv420p {final_output}"
    )
    os.remove(temp_output)
    
    print("Video ready:")
    display(Video(final_output, embed=True, width=600, html_attributes="controls autoplay loop"))
else:
    print("Error: Output video not generated.")

# Load models
vehicle_model = YOLO("/kaggle/input/best-model1/other/best-model/1/best.pt")


#vehicle track and count

MODEL_PATH = '/kaggle/input/best-model1/other/best-model/1/best.pt'
VIDEO_PATH = 'https://docs.google.com/uc?export=download&confirm=&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-'
OUTPUT_PATH = 'LaneCountFinalResult.mp4'
TMP_OUTPUT_PATH = 'tmp_output.mp4'

# Vehicle classes to detect (all vehicle types from your dataset)
VEHICLE_CLASSES = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


# Detection parameters
DETECTION_CONFIDENCE = 0.35 
SCALE_PERCENT = 50

# Tracking parameters
IOU_THRESHOLD = 0.4
MAX_AGE = 30

# Counting line parameters
LINE_POSITION_PERCENT = 65  
LANE_SPLIT_PERCENT = 50     

# Sample frames
SAMPLE_FRAMES = [28, 29, 32, 40, 42, 50, 58]


def resize_frame(frame, scale_percent):
    """Resize frame by percentage."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

class DirectionalCounter:
    """
    Counts vehicles crossing a single line.
    Left lane = IN
    Right lane = OUT
    """
    
    def __init__(self, counting_line_y, lane_split_x):
        self.counting_line_y = counting_line_y
        self.lane_split_x = lane_split_x
        
        # Track vehicle states
        self.vehicles = {}
        
        # Counted sets
        self.counted_in = set()
        self.counted_out = set()
        
        # Counters
        self.count_in = 0
        self.count_out = 0
        self.class_count_in = defaultdict(int)
        self.class_count_out = defaultdict(int)
        
        # Statistics
        self.total_detections = 0
        self.active_tracks = set()
        
    def update(self, track_id, center_x, center_y, class_id, class_name):
        """Update vehicle and check if it crossed the counting line."""
        
        self.total_detections += 1
        self.active_tracks.add(track_id)
        
        # Initialize new vehicle
        if track_id not in self.vehicles:
            self.vehicles[track_id] = {
                'class_id': class_id,
                'class_name': class_name,
                'prev_y': center_y,
                'counted': False,
                'direction': None
            }
        
        vehicle = self.vehicles[track_id]
        prev_y = vehicle['prev_y']
        curr_y = center_y
        
        # Update position
        vehicle['prev_y'] = curr_y
        
        # Already counted
        if vehicle['counted']:
            return None
        
        # Check if crossed the line
        line_y = self.counting_line_y
        crossed = (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y)
        
        if not crossed:
            return None
        
        # Determine lane
        lane = 'LEFT' if center_x < self.lane_split_x else 'RIGHT'
        
        # LEFT LANE = IN
        if lane == 'LEFT':
            if track_id not in self.counted_in:
                self.counted_in.add(track_id)
                vehicle['counted'] = True
                vehicle['direction'] = 'IN'
                self.count_in += 1
                self.class_count_in[class_name] += 1
                print(f"IN: ID {track_id} ({class_name}) at x={center_x}, y={curr_y}")
                return 'IN'
        
        # RIGHT LANE = OUT
        else:
            if track_id not in self.counted_out:
                self.counted_out.add(track_id)
                vehicle['counted'] = True
                vehicle['direction'] = 'OUT'
                self.count_out += 1
                self.class_count_out[class_name] += 1
                print(f"OUT: ID {track_id} ({class_name}) at x={center_x}, y={curr_y}")
                return 'OUT'
        
        return None
    
    def get_vehicle_color(self, track_id):
        """Get color based on direction."""
        if track_id not in self.vehicles:
            return (255, 255, 255)
        
        vehicle = self.vehicles[track_id]
        if vehicle['direction'] == 'IN':
            return (0, 255, 0)  # Green
        elif vehicle['direction'] == 'OUT':
            return (0, 0, 255)  # Red
        else:
            return (255, 255, 0)  # Yellow
    
    def cleanup_old_tracks(self):
        """Remove old tracks."""
        to_remove = [tid for tid, v in self.vehicles.items() 
                    if tid not in self.active_tracks]
        for tid in to_remove:
            del self.vehicles[tid]
        self.active_tracks.clear()

# MAIN PROCESSING FUNCTION

def process_video():
    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)
    class_names = model.model.names
    print(f"Model loaded")
    print(f"Detection Confidence: {DETECTION_CONFIDENCE}")
    
    print(f"\nOpening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate scaled dimensions
    width = int(original_width * SCALE_PERCENT / 100)
    height = int(original_height * SCALE_PERCENT / 100)
    print(f"Resolution: {width}x{height}")
    
    # Calculate counting line and lane split
    counting_line_y = int(height * LINE_POSITION_PERCENT / 100)
    lane_split_x = int(width * LANE_SPLIT_PERCENT / 100)
    
    print(f"\nConfiguration:")
    print(f"   Counting Line Y: {counting_line_y} ({LINE_POSITION_PERCENT}%)")
    print(f"   Lane Split X: {lane_split_x}")
    print(f"   Left Lane: IN")
    print(f"   Right Lane: OUT")
    
    # Initialize counter
    counter = DirectionalCounter(counting_line_y, lane_split_x)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TMP_OUTPUT_PATH, fourcc, fps, (width, height))
    
    frames_list = []
    frame_count = 0
    
    print(f"\nProcessing {total_frames} frames...")
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = resize_frame(frame, SCALE_PERCENT)
        
        # Run YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=DETECTION_CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=VEHICLE_CLASSES,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        # Draw counting line (YELLOW)
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 255, 255), 5)
        
        # Draw lane separator (MAGENTA)
        cv2.line(frame, (lane_split_x, 0), (lane_split_x, height), (255, 0, 255), 3)
        
        # Process vehicle detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate center (bottom-center)
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                
                class_name = class_names[cls_id]
                
                # Update counter
                direction = counter.update(track_id, center_x, center_y, cls_id, class_name)
                
                # Get color
                color = counter.get_vehicle_color(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Create simple label with ID and confidence
                label = f"ID:{track_id} {conf:.2f}"
                
                # Draw label
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw simple counter panel
        panel_height = 70
        cv2.rectangle(frame, (10, 10), (250, panel_height + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, panel_height + 10), (255, 255, 255), 2)
        
        # Display counters - ONLY "IN" and "OUT"
        cv2.putText(frame, f"IN: {counter.count_in}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {counter.count_out}",
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Cleanup old tracks periodically
        if frame_count % 30 == 0:
            counter.cleanup_old_tracks()
        
        # Save frame
        out.write(frame)
        
        # Capture samples
        if frame_count in SAMPLE_FRAMES:
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        pbar.update(1)
        frame_count += 1
    
    pbar.close()
    cap.release()
    out.release()
    
    # Print summary
    print("\n" + "="*70)
    print("VEHICLE COUNTING SUMMARY")
    print("="*70)
    
    print(f"\nIN: {counter.count_in} vehicles")
    if counter.class_count_in:
        for class_name, count in sorted(counter.class_count_in.items()):
            print(f"   - {class_name}: {count}")
    
    print(f"\nOUT: {counter.count_out} vehicles")
    if counter.class_count_out:
        for class_name, count in sorted(counter.class_count_out.items()):
            print(f"   - {class_name}: {count}")
    
    print(f"\nTOTAL: {counter.count_in + counter.count_out} vehicles")
    print(f"   Net: {counter.count_in - counter.count_out} (IN - OUT)")
    print("="*70)
    
    # Re-encode video
    print("\nRe-encoding video...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    subprocess.run([
        "ffmpeg", "-i", TMP_OUTPUT_PATH,
        "-vcodec", "libx264", "-crf", "18",
        "-preset", "veryfast",
        "-hide_banner", "-loglevel", "error",
        OUTPUT_PATH
    ], capture_output=True)
    
    if os.path.exists(TMP_OUTPUT_PATH):
        os.remove(TMP_OUTPUT_PATH)
    
    print("Video complete!")
    
    # Display samples
    if frames_list:
        print(f"\nSample Frames:")
        for idx, frame_img in enumerate(frames_list):
            plt.figure(figsize=(14, 10))
            plt.imshow(frame_img)
            plt.title(f"Frame {SAMPLE_FRAMES[idx]}", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    print("\nOutput Video:")
    display(Video(OUTPUT_PATH, width=640, height=360, embed=True))
    
    return counter

# EXECUTE

if __name__ == "__main__":
    print("="*70)
    print("VEHICLE COUNTING SYSTEM")
    print("="*70)
    print("Left Lane: IN")
    print("Right Lane: OUT")
    print(f"Confidence: {DETECTION_CONFIDENCE}")
    print("="*70 + "\n")
    
    counter = process_video()
    print("\nDone!")

#Red Light Violation
# CONFIGURATION
MODEL_PATH = "/kaggle/input/best-model1/other/best-model/1/best.pt"
VIDEO_PATH = "/kaggle/input/rltesting/redLightV2.mp4"
OUTPUT_PATH = "ViolationDetect2.mp4"
TMP_OUTPUT_PATH = "temp_output.mp4" 

# Class Mapping
VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] 

RED_LIGHT_CLASS = 6 
GREEN_LIGHT_CLASS = 7
YELLOW_LIGHT_CLASS = 8

# Stop Line Configuration (horizontal line)
STOP_LINE = [(10, 170), (340, 170)] 

# Detection Thresholds
INFERENCE_CONF_THRESHOLD = 0.05
LIGHT_CONF_THRESHOLD = 0.05
VEHICLE_CONF_THRESHOLD = 0.35

# Sample frames for visualization
SAMPLE_FRAME_INDICES = [28, 29, 32, 40, 42, 50, 58]

# Light persistence (frames to remember light state)
MAX_LIGHT_PERSISTENCE = 30

# HELPER FUNCTIONS

def get_line_y_at_x(x, p1, p2):
    """Calculate Y coordinate on the line at given X."""
    x1, y1 = p1
    x2, y2 = p2
    
    if x2 == x1:
        return y1
    
    slope = (y2 - y1) / (x2 - x1)
    y = y1 + slope * (x - x1)
    return int(y)

def is_vehicle_crossing_line(prev_center, curr_center, line_points, tolerance=15):
    """
    Detect if vehicle crossed or is touching the stop line.
    Returns True when vehicle crosses from above to at/below the line.
    """
    if prev_center is None or curr_center is None:
        return False
    
    prev_x, prev_y = prev_center
    curr_x, curr_y = curr_center
    
    p1, p2 = line_points
    
    # Get line Y at both previous and current X positions
    line_y_prev = get_line_y_at_x(prev_x, p1, p2)
    line_y_curr = get_line_y_at_x(curr_x, p1, p2)
    
    # Crossing detection: vehicle moved from above line to touching/crossing it
    # Increased tolerance to detect even slight crossing
    was_above = prev_y < (line_y_prev - tolerance)
    is_at_or_below = curr_y >= (line_y_curr - tolerance)
    
    if was_above and is_at_or_below:
        return True
    
    return False

def is_vehicle_in_violation_zone(center, line_points, tolerance=10):
    """
    Check if vehicle is touching or past the stop line.
    Used for continuous violation checking during red light.
    """
    if center is None:
        return False
    
    center_x, center_y = center
    p1, p2 = line_points
    
    # Get line Y at this X
    line_y = get_line_y_at_x(center_x, p1, p2)
    
    # Check if vehicle is at or past the line
    if center_y >= (line_y - tolerance):
        return True
    
    return False

def draw_tracking_info(frame, track_id, x1, y1, x2, y2, is_violation, model_names, cls_id):
    """Draw bounding box and label for tracked vehicle."""
    if is_violation:
        color = (0, 0, 255)  # Red
        label = f"VIOLATION ID:{track_id}"
        thickness = 2
    else:
        color = (0, 255, 0)  # Green
        label = f"{model_names[cls_id]} ID:{track_id}"
        thickness = 1
    
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label background
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
    
    # Label text
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Center point
    center_x = int((x1 + x2) / 2)
    center_y = int(y2)
    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 255), -1)


# MAIN PROCESSING FUNCTION

def process_video():
    print(" Loading YOLO Model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f" Model Loaded. Classes: {model.names}")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(" Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f" Video: {width}x{height} @ {fps}fps, Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TMP_OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    violated_vehicle_ids = set()  # Vehicles that violated red light
    track_history = {}  # Previous positions: {track_id: (x, y)}
    vehicle_violation_frames = {}  # Track when each vehicle was first seen violating
    captured_samples = {}
    
    # Light state
    current_light_status = "OFF"
    current_light_color = (128, 128, 128)
    last_detected_light = "OFF"
    frames_since_light_seen = 999

    print(f" Starting video processing...")
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=INFERENCE_CONF_THRESHOLD, verbose=False)
      
        # DETECT TRAFFIC LIGHT STATUS
        light_detected_this_frame = False
        temp_light_status = "OFF"
        temp_light_color = (128, 128, 128)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < LIGHT_CONF_THRESHOLD:
                    continue
                
                if cls_id == RED_LIGHT_CLASS:
                    temp_light_status = "RED"
                    temp_light_color = (0, 0, 255)
                    light_detected_this_frame = True
                    last_detected_light = "RED"
                    frames_since_light_seen = 0
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "RED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)
                    
                elif cls_id == GREEN_LIGHT_CLASS:
                    temp_light_status = "GREEN"
                    temp_light_color = (0, 255, 0)
                    light_detected_this_frame = True
                    last_detected_light = "GREEN"
                    frames_since_light_seen = 0
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "GREEN", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)
                    
                elif cls_id == YELLOW_LIGHT_CLASS:
                    temp_light_status = "YELLOW"
                    temp_light_color = (0, 255, 255)
                    light_detected_this_frame = True
                    last_detected_light = "YELLOW"
                    frames_since_light_seen = 0
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "YELLOW", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)
        
        # Apply light persistence
        if light_detected_this_frame:
            current_light_status = temp_light_status
            current_light_color = temp_light_color
        elif frames_since_light_seen < MAX_LIGHT_PERSISTENCE:
            current_light_status = last_detected_light
            if last_detected_light == "RED":
                current_light_color = (0, 0, 255)
            elif last_detected_light == "GREEN":
                current_light_color = (0, 255, 0)
            elif last_detected_light == "YELLOW":
                current_light_color = (0, 255, 255)
            frames_since_light_seen += 1
        else:
            current_light_status = "OFF"
            current_light_color = (128, 128, 128)
            frames_since_light_seen += 1
        
        # Check if red light is active
        is_red_light = (current_light_status == "RED")
        
        # STEP 2: TRACK VEHICLES AND DETECT VIOLATIONS
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < VEHICLE_CONF_THRESHOLD:
                    continue
                
                if cls_id in VEHICLE_CLASSES and box.id is not None:
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate bottom-center of vehicle (best for crossing detection)
                    center_x = int((x1 + x2) / 2)
                    center_y = int(y2)
                    current_center = (center_x, center_y)
                    
                    # Check if this vehicle already violated (don't recheck)
                    already_violated = track_id in violated_vehicle_ids
                    
                    if not already_violated:
                        # METHOD 1: Check if vehicle is crossing the line RIGHT NOW
                        if track_id in track_history:
                            prev_center = track_history[track_id]
                            
                            # Detect if vehicle crossed/touched the line THIS frame
                            just_crossed = is_vehicle_crossing_line(prev_center, current_center, STOP_LINE)
                            
                            # VIOLATION: Crossed while red light is ON
                            if just_crossed and is_red_light:
                                violated_vehicle_ids.add(track_id)
                                vehicle_violation_frames[track_id] = frame_count
                                print(f" VIOLATION DETECTED - CROSSING!")
                                print(f"   Frame: {frame_count}")
                                print(f"   Vehicle ID: {track_id}")
                                print(f"   Previous Y: {prev_center[1]} -> Current Y: {current_center[1]}")
                                print("-" * 50)
                        
                        # METHOD 2: Check if vehicle is IN violation zone during red
                        # (catches vehicles that are already touching/past the line)
                        if is_red_light and is_vehicle_in_violation_zone(current_center, STOP_LINE):
                            # Only mark as violation if we have tracking history
                            # This prevents false positives for vehicles already past line when tracking starts
                            if track_id in track_history:
                                prev_center = track_history[track_id]
                                # Check if vehicle was approaching from above (not already past)
                                line_y = get_line_y_at_x(prev_center[0], STOP_LINE[0], STOP_LINE[1])
                                if prev_center[1] < line_y + 20:  # Was recently before/at the line
                                    if track_id not in violated_vehicle_ids:
                                        violated_vehicle_ids.add(track_id)
                                        vehicle_violation_frames[track_id] = frame_count
                                        print(f" VIOLATION DETECTED - IN ZONE!")
                                        print(f"   Frame: {frame_count}")
                                        print(f"   Vehicle ID: {track_id}")
                                        print(f"   Position Y: {current_center[1]}")
                                        print("-" * 50)
                    
                    # Update position history
                    track_history[track_id] = current_center
                    
                    # Draw vehicle with violation status
                    is_violator = (track_id in violated_vehicle_ids)
                    draw_tracking_info(frame, track_id, x1, y1, x2, y2, is_violator, model.names, cls_id)
        
        # STEP 3: DRAW UI ELEMENTS
        
        # Draw stop line
        cv2.line(frame, STOP_LINE[0], STOP_LINE[1], (0, 255, 255), 2)
        cv2.putText(frame, "STOP LINE", (STOP_LINE[0][0], STOP_LINE[0][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Status panel
        cv2.rectangle(frame, (10, 10), (250, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 85), current_light_color, 1)
        
        cv2.putText(frame, f"Traffic Light: {current_light_status}", 
                   (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_light_color, 1)
        
        violation_count = len(violated_vehicle_ids)
        violation_display_color = (0, 0, 255) if violation_count > 0 else (255, 255, 255)
        cv2.putText(frame, f"Violations: {violation_count}", 
                   (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, violation_display_color, 1)
        
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Capture samples
        if frame_count in SAMPLE_FRAME_INDICES:
            captured_samples[frame_count] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        out.write(frame)
        pbar.update(1)
        frame_count += 1

    pbar.close()
    cap.release()
    out.release()

    print(f"\n Processing Complete!")
    print(f" Total Violations: {len(violated_vehicle_ids)}")
    if violated_vehicle_ids:
        print(f" Violated Vehicle IDs: {sorted(violated_vehicle_ids)}")
    else:
        print(" No violations detected!")

    print("\n Re-encoding video...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    subprocess.run([
        "ffmpeg", "-i", TMP_OUTPUT_PATH, 
        "-vcodec", "libx264", "-crf", "18", 
        "-preset", "veryfast", 
        "-hide_banner", "-loglevel", "error", 
        OUTPUT_PATH
    ], capture_output=True)
    
    if os.path.exists(TMP_OUTPUT_PATH):
        os.remove(TMP_OUTPUT_PATH)
    
    print("✅ Video encoding complete!")

    # Display samples
    if captured_samples:
        print("\n Sample Frames:")
        fig, axes = plt.subplots(len(SAMPLE_FRAME_INDICES), 1, figsize=(12, 4*len(SAMPLE_FRAME_INDICES)))
        if len(SAMPLE_FRAME_INDICES) == 1:
            axes = [axes]
        
        for idx, frame_idx in enumerate(SAMPLE_FRAME_INDICES):
            if frame_idx in captured_samples:
                axes[idx].imshow(captured_samples[frame_idx])
                axes[idx].set_title(f"Frame {frame_idx}", fontsize=14, fontweight='bold')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\n Output Video:")
    display(Video(OUTPUT_PATH, width=800))
    
    return violated_vehicle_ids

# EXECUTE

if __name__ == "__main__":
    violations = process_video()
    print(f"\n{'='*60}")
    print(f" FINAL REPORT")
    print(f"{'='*60}")
    print(f"Total Violations: {len(violations)}")
    if violations:
        print(f"Violated Vehicle IDs: {sorted(violations)}")
    else:
        print("No red light violations detected.")
    print(f"{'='*60}")

# #merging 
# CONFIGURATION

MODEL_PATH = '/kaggle/input/best-model1/other/best-model/1/best.pt'
VIDEO_PATH = 'https://docs.google.com/uc?export=download&confirm=&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-'
OUTPUT_PATH = 'MergedLaneCountViolation.mp4'
TMP_OUTPUT_PATH = 'tmp_output.mp4'

# All relevant classes
VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
RED_LIGHT_CLASS = 6
GREEN_LIGHT_CLASS = 7
YELLOW_LIGHT_CLASS = 8

# Detection parameters
DETECTION_CONFIDENCE = 0.35
LIGHT_CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.4
MAX_AGE = 30

# Single stop/counting line parameters (horizontal line across frame)
LINE_POSITION_PERCENT = 65  # Position of the single stop/counting line 
LANE_SPLIT_PERCENT = 50  # Divides left (IN) from right (OUT)

# Light persistence (frames to remember light state)
MAX_LIGHT_PERSISTENCE = 30

# Sample frames for visualization
SAMPLE_FRAMES = [28, 29, 32, 40, 42, 50, 58]

# HELPER FUNCTIONS
def get_line_y_at_x(x, p1, p2):
    """Calculate Y coordinate on the line at given X."""
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    y = y1 + slope * (x - x1)
    return int(y)

def is_vehicle_crossing_line(prev_center, curr_center, line_points, tolerance=15):
    """Detect if vehicle crossed or is touching the stop line."""
    if prev_center is None or curr_center is None:
        return False
    prev_x, prev_y = prev_center
    curr_x, curr_y = curr_center
    p1, p2 = line_points
    line_y_prev = get_line_y_at_x(prev_x, p1, p2)
    line_y_curr = get_line_y_at_x(curr_x, p1, p2)
    was_above = prev_y < (line_y_prev - tolerance)
    is_at_or_below = curr_y >= (line_y_curr - tolerance)
    if was_above and is_at_or_below:
        return True
    return False

def is_vehicle_in_violation_zone(center, line_points, tolerance=10):
    """Check if vehicle is touching or past the stop line."""
    if center is None:
        return False
    center_x, center_y = center
    p1, p2 = line_points
    line_y = get_line_y_at_x(center_x, p1, p2)
    if center_y >= (line_y - tolerance):
        return True
    return False

class DirectionalCounter:
    """Counts vehicles crossing the single line. Left lane = IN, Right lane = OUT."""
    def __init__(self, counting_line_y, lane_split_x):
        self.counting_line_y = counting_line_y
        self.lane_split_x = lane_split_x
        self.vehicles = {}
        self.counted_in = set()
        self.counted_out = set()
        self.count_in = 0
        self.count_out = 0
        self.class_count_in = defaultdict(int)
        self.class_count_out = defaultdict(int)
        self.total_detections = 0
        self.active_tracks = set()

    def update(self, track_id, center_x, center_y, class_id, class_name):
        """Update vehicle and check if it crossed the counting line."""
        self.total_detections += 1
        self.active_tracks.add(track_id)
        if track_id not in self.vehicles:
            self.vehicles[track_id] = {
                'class_id': class_id,
                'class_name': class_name,
                'prev_y': center_y,
                'counted': False,
                'direction': None
            }
        vehicle = self.vehicles[track_id]
        prev_y = vehicle['prev_y']
        curr_y = center_y
        vehicle['prev_y'] = curr_y
        if vehicle['counted']:
            return None
        line_y = self.counting_line_y
        crossed = (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y)
        if not crossed:
            return None
        lane = 'LEFT' if center_x < self.lane_split_x else 'RIGHT'
        if lane == 'LEFT':
            if track_id not in self.counted_in:
                self.counted_in.add(track_id)
                vehicle['counted'] = True
                vehicle['direction'] = 'IN'
                self.count_in += 1
                self.class_count_in[class_name] += 1
                print(f"IN: ID {track_id} ({class_name}) at x={center_x}, y={curr_y}")
                return 'IN'
        else:
            if track_id not in self.counted_out:
                self.counted_out.add(track_id)
                vehicle['counted'] = True
                vehicle['direction'] = 'OUT'
                self.count_out += 1
                self.class_count_out[class_name] += 1
                print(f"OUT: ID {track_id} ({class_name}) at x={center_x}, y={curr_y}")
                return 'OUT'
        return None

    def get_vehicle_color(self, track_id):
        """Get color based on direction."""
        if track_id not in self.vehicles:
            return (255, 255, 255)
        vehicle = self.vehicles[track_id]
        if vehicle['direction'] == 'IN':
            return (0, 255, 0)  # Green
        elif vehicle['direction'] == 'OUT':
            return (0, 0, 255)  # Red
        else:
            return (255, 255, 0)  # Yellow

    def cleanup_old_tracks(self):
        """Remove old tracks."""
        to_remove = [tid for tid, v in self.vehicles.items() if tid not in self.active_tracks]
        for tid in to_remove:
            del self.vehicles[tid]
        self.active_tracks.clear()

def draw_tracking_info(frame, track_id, x1, y1, x2, y2, is_violation, direction_color, model_names, cls_id):
    """Draw bounding box and label for tracked vehicle."""
    if is_violation:
        color = (0, 0, 255)  # Red override for violation
        label = f"VIOLATION ID:{track_id} {model_names[cls_id]}"
        thickness = 4
    else:
        color = direction_color
        label = f"{model_names[cls_id]} ID:{track_id}"
        thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    center_x = int((x1 + x2) / 2)
    center_y = int(y2)
    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 255), -1)

# MAIN PROCESSING FUNCTION
def process_video():
    print("Loading YOLO Model...")
    model = YOLO(MODEL_PATH)
    class_names = model.model.names
    print(f"Model loaded. Classes: {class_names}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, Frames: {total_frames}")

    # Calculate single stop/counting line and lane split
    stop_line_y = int(height * LINE_POSITION_PERCENT / 100)
    lane_split_x = int(width * LANE_SPLIT_PERCENT / 100)
    STOP_LINE = [(0, stop_line_y), (width, stop_line_y)]

    print(f"\nConfiguration:")
    print(f" Stop/Counting Line Y: {stop_line_y} ({LINE_POSITION_PERCENT}%)")
    print(f" Lane Split X: {lane_split_x}")
    print(f" Left Lane: IN")
    print(f" Right Lane: OUT")

    # Initialize counter and violation tracking
    counter = DirectionalCounter(stop_line_y, lane_split_x)
    violated_vehicle_ids = set()
    track_history = {}
    vehicle_violation_frames = {}
    captured_samples = {}

    # Light state
    current_light_status = "OFF"
    current_light_color = (128, 128, 128)
    last_detected_light = "OFF"
    frames_since_light_seen = 999

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TMP_OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    print(f"\nProcessing {total_frames} frames...")
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking
        results = model.track(
            frame,
            persist=True,
            conf=DETECTION_CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        # Draw single stop/counting line (YELLOW) and lane separator (MAGENTA)
        cv2.line(frame, STOP_LINE[0], STOP_LINE[1], (0, 255, 255), 5)
        cv2.line(frame, (lane_split_x, 0), (lane_split_x, height), (255, 0, 255), 3)

        # STEP 1: DETECT TRAFFIC LIGHT STATUS
        light_detected_this_frame = False
        temp_light_status = "OFF"
        temp_light_color = (128, 128, 128)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confidences):
                if conf < LIGHT_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if cls_id == RED_LIGHT_CLASS:
                    temp_light_status = "RED"
                    temp_light_color = (0, 0, 255)
                    light_detected_this_frame = True
                    last_detected_light = "RED"
                    frames_since_light_seen = 0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "RED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)
                elif cls_id == GREEN_LIGHT_CLASS:
                    temp_light_status = "GREEN"
                    temp_light_color = (0, 255, 0)
                    light_detected_this_frame = True
                    last_detected_light = "GREEN"
                    frames_since_light_seen = 0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "GREEN", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)
                elif cls_id == YELLOW_LIGHT_CLASS:
                    temp_light_status = "YELLOW"
                    temp_light_color = (0, 255, 255)
                    light_detected_this_frame = True
                    last_detected_light = "YELLOW"
                    frames_since_light_seen = 0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), temp_light_color, 2)
                    cv2.putText(frame, "YELLOW", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_light_color, 1)

        # Apply light persistence
        if light_detected_this_frame:
            current_light_status = temp_light_status
            current_light_color = temp_light_color
        elif frames_since_light_seen < MAX_LIGHT_PERSISTENCE:
            current_light_status = last_detected_light
            if last_detected_light == "RED":
                current_light_color = (0, 0, 255)
            elif last_detected_light == "GREEN":
                current_light_color = (0, 255, 0)
            elif last_detected_light == "YELLOW":
                current_light_color = (0, 255, 255)
            frames_since_light_seen += 1
        else:
            current_light_status = "OFF"
            current_light_color = (128, 128, 128)
            frames_since_light_seen += 1

        is_red_light = (current_light_status == "RED")

        # STEP 2: PROCESS VEHICLES FOR COUNTING AND VIOLATIONS
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box)
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                current_center = (center_x, center_y)
                class_name = class_names[cls_id]

                # Update counting (always)
                direction = counter.update(track_id, center_x, center_y, cls_id, class_name)

                # Check violation (only if not already violated)
                already_violated = track_id in violated_vehicle_ids
                if not already_violated:
                    if track_id in track_history:
                        prev_center = track_history[track_id]
                        just_crossed = is_vehicle_crossing_line(prev_center, current_center, STOP_LINE)
                        if just_crossed and is_red_light:
                            violated_vehicle_ids.add(track_id)
                            vehicle_violation_frames[track_id] = frame_count
                            print(f"VIOLATION DETECTED - CROSSING! ID: {track_id} Frame: {frame_count}")
                    if is_red_light and is_vehicle_in_violation_zone(current_center, STOP_LINE):
                        if track_id in track_history:
                            prev_center = track_history[track_id]
                            line_y = get_line_y_at_x(prev_center[0], STOP_LINE[0], STOP_LINE[1])
                            if prev_center[1] < line_y + 20:
                                if track_id not in violated_vehicle_ids:
                                    violated_vehicle_ids.add(track_id)
                                    vehicle_violation_frames[track_id] = frame_count
                                    print(f"VIOLATION DETECTED - IN ZONE! ID: {track_id} Frame: {frame_count}")

                # Update position history
                track_history[track_id] = current_center

                # Get direction color
                direction_color = counter.get_vehicle_color(track_id)

                # Draw with violation status
                is_violator = (track_id in violated_vehicle_ids)
                draw_tracking_info(frame, track_id, x1, y1, x2, y2, is_violator, direction_color, class_names, cls_id)

        # Draw merged status panel
        panel_height = 110
        cv2.rectangle(frame, (10, 10), (250, panel_height + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, panel_height + 10), current_light_color, 2)
        cv2.putText(frame, f"IN: {counter.count_in}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {counter.count_out}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        violation_count = len(violated_vehicle_ids)
        violation_color = (0, 0, 255) if violation_count > 0 else (255, 255, 255)
        cv2.putText(frame, f"Violations: {violation_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, violation_color, 2)
        cv2.putText(frame, f"Light: {current_light_status}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_light_color, 2)

        # Cleanup old tracks periodically
        if frame_count % 30 == 0:
            counter.cleanup_old_tracks()

        # Capture samples
        if frame_count in SAMPLE_FRAMES:
            captured_samples[frame_count] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out.write(frame)
        pbar.update(1)
        frame_count += 1

    pbar.close()
    cap.release()
    out.release()

    # Print summary
    print("\n" + "="*70)
    print("MERGED VEHICLE COUNTING AND VIOLATION SUMMARY")
    print("="*70)
    print(f"\nIN: {counter.count_in} vehicles")
    for class_name, count in sorted(counter.class_count_in.items()):
        print(f" - {class_name}: {count}")
    print(f"\nOUT: {counter.count_out} vehicles")
    for class_name, count in sorted(counter.class_count_out.items()):
        print(f" - {class_name}: {count}")
    print(f"\nTOTAL: {counter.count_in + counter.count_out} vehicles")
    print(f"Net: {counter.count_in - counter.count_out} (IN - OUT)")
    print(f"\nTotal Violations: {len(violated_vehicle_ids)}")
    if violated_vehicle_ids:
        print(f"Violated Vehicle IDs: {sorted(violated_vehicle_ids)}")
    print("="*70)

    # Re-encode video
    print("\nRe-encoding video...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    subprocess.run([
        "ffmpeg", "-i", TMP_OUTPUT_PATH,
        "-vcodec", "libx264", "-crf", "18",
        "-preset", "veryfast",
        "-hide_banner", "-loglevel", "error",
        OUTPUT_PATH
    ], capture_output=True)
    if os.path.exists(TMP_OUTPUT_PATH):
        os.remove(TMP_OUTPUT_PATH)
    print("Video complete!")

    # Display samples
    if captured_samples:
        print(f"\nSample Frames:")
        fig, axes = plt.subplots(len(SAMPLE_FRAMES), 1, figsize=(12, 4*len(SAMPLE_FRAMES)))
        if len(SAMPLE_FRAMES) == 1:
            axes = [axes]
        for idx, frame_idx in enumerate(SAMPLE_FRAMES):
            if frame_idx in captured_samples:
                axes[idx].imshow(captured_samples[frame_idx])
                axes[idx].set_title(f"Frame {frame_idx}", fontsize=14, fontweight='bold')
                axes[idx].axis('off')
        plt.tight_layout()
        plt.show()

    print("\nOutput Video:")
    display(Video(OUTPUT_PATH, width=800, embed=True))

    return counter, violated_vehicle_ids

# EXECUTE

if __name__ == "__main__":
    print("="*70)
    print("MERGED VEHICLE COUNTING AND RED LIGHT VIOLATION SYSTEM")
    print("="*70)
    print("Left Lane: IN")
    print("Right Lane: OUT")
    print(f"Confidence: {DETECTION_CONFIDENCE}")
    print("="*70 + "\n")
    counter, violations = process_video()
    print("\nDone!")

