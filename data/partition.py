import os
import random
from collections import defaultdict

def get_all_images(data_dir):
    """
    Scans the train folder and returns a dict:
    { 'NORMAL': [list of file paths], 'PNEUMONIA': [list of file paths] }
    """
    class_images = defaultdict(list)
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(class_dir, fname)
                class_images[class_name].append(full_path)
    
    # Shuffle for randomness
    for cls in class_images:
        random.shuffle(class_images[cls])
    
    return class_images


def partition_into_hospitals(data_dir, seed=42):
    """
    Partitions training data into 5 hospitals with non-IID distribution.
    
    Expert hospitals:   more data, balanced classes
    Non-expert hospitals: less data, skewed classes
    
    Returns:
        list of dicts, one per hospital:
        [
            {'hospital_id': 0, 'is_expert': True, 'images': [...], 'labels': [...]},
            ...
        ]
    """
    random.seed(seed)
    
    class_images = get_all_images(data_dir)
    normal_imgs   = class_images['NORMAL']     # 1349 images
    pneumonia_imgs = class_images['PNEUMONIA'] # 3883 images
    
    # ----------------------------
    # Define hospital partitions
    # ----------------------------
    # Each entry: (num_normal, num_pneumonia, is_expert)
    hospital_configs = [
        (400, 700, True),   # Hospital 0 - Expert,     1100 imgs, balanced-ish
        (350, 600, True),   # Hospital 1 - Expert,     950 imgs,  balanced-ish
        (100, 300, False),  # Hospital 2 - Non-Expert, 400 imgs,  pneumonia-heavy
        (180,  80, False),  # Hospital 3 - Non-Expert, 260 imgs,  normal-heavy
        ( 60, 120, False),  # Hospital 4 - Non-Expert, 180 imgs,  very limited
    ]
    
    hospitals = []
    normal_idx    = 0
    pneumonia_idx = 0
    
    for h_id, (n_normal, n_pneumonia, is_expert) in enumerate(hospital_configs):
        
        # Slice images for this hospital
        h_normal    = normal_imgs[normal_idx    : normal_idx    + n_normal]
        h_pneumonia = pneumonia_imgs[pneumonia_idx : pneumonia_idx + n_pneumonia]
        
        # Combine images and labels (0 = Normal, 1 = Pneumonia)
        images = h_normal + h_pneumonia
        labels = [0] * len(h_normal) + [1] * len(h_pneumonia)
        
        # Shuffle combined list
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)
        
        hospitals.append({
            'hospital_id': h_id,
            'is_expert'  : is_expert,
            'images'     : list(images),
            'labels'     : list(labels),
            'num_normal' : n_normal,
            'num_pneumonia': n_pneumonia,
            'total'      : n_normal + n_pneumonia
        })
        
        normal_idx    += n_normal
        pneumonia_idx += n_pneumonia
    
    return hospitals

def get_reference_dataset(data_dir, num_images=200, seed=42):
    """
    Returns a small shared reference dataset used for knowledge sharing.
    These are images (no labels needed) that expert hospitals run inference
    on to generate soft predictions for non-expert hospitals.
    
    Uses images NOT assigned to any hospital partition.
    """
    random.seed(seed)
    
    class_images = get_all_images(data_dir)
    normal_imgs    = class_images['NORMAL']
    pneumonia_imgs = class_images['PNEUMONIA']
    
    # Hospital partition uses up to index:
    # Normal:    400+350+100+180+60 = 1090
    # Pneumonia: 700+600+300+80+120 = 1800
    # So we take from the remaining images
    
    remaining_normal    = normal_imgs[1090:]   # 1349-1090 = 259 remaining
    remaining_pneumonia = pneumonia_imgs[1800:] # 3883-1800 = 2083 remaining
    
    # Take 100 from each class for balance
    ref_normal    = remaining_normal[:100]
    ref_pneumonia = remaining_pneumonia[:100]
    
    ref_images = ref_normal + ref_pneumonia
    ref_labels = [0]*100 + [1]*100
    
    # Shuffle
    combined = list(zip(ref_images, ref_labels))
    random.shuffle(combined)
    ref_images, ref_labels = zip(*combined)
    
    return {
        'images': list(ref_images),
        'labels': list(ref_labels),
        'total' : len(ref_images)
    }

if __name__ == "__main__":
    # Quick test - update this path to your actual data path
    DATA_DIR = r"G:\PROJECTS\FL with KD\Kaggle (Pneumonia) Chest X-ray Dataset\chest_xray\train"
    
    hospitals = partition_into_hospitals(DATA_DIR)
    
    print("=" * 50)
    print("HOSPITAL DATA DISTRIBUTION")
    print("=" * 50)
    for h in hospitals:
        expert_tag = "EXPERT    " if h['is_expert'] else "NON-EXPERT"
        print(f"Hospital {h['hospital_id']} [{expert_tag}] | "
              f"Total: {h['total']:4d} | "
              f"Normal: {h['num_normal']:4d} | "
              f"Pneumonia: {h['num_pneumonia']:4d}")
    print("=" * 50)
    print(f"Total images assigned: {sum(h['total'] for h in hospitals)}")
