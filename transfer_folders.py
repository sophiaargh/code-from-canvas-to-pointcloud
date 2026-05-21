import shutil
import os
from pathlib import Path
import sys

# Read style from command line argument, default to "engraving" if none provided
style = sys.argv[1] if len(sys.argv) > 1 else "engraving"

# Configuration
src_root = Path("/scratch/izar/silly/BlendedMVS/renamed/")
dst_root = Path(f"/scratch/izar/silly/BlendedMVS/telestyle_output/{style}/")
folders_to_copy = ["cams", "rendered_depth_maps"]

def set_permissions_recursive(path):
    """Recursively sets permissions to 777 for a directory and all its contents."""
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)
    os.chmod(path, 0o777) # Set the root folder itself

def sync_folders():
    any_copied = False  # Track if any copy action actually happened

    for scene_dir in src_root.glob("scene_*"):
        if not scene_dir.is_dir():
            continue
            
        scene_name = scene_dir.name
        target_scene_dir = dst_root / scene_name
        
        if target_scene_dir.exists():
            scene_header_printed = False
            
            for folder in folders_to_copy:
                src_path = scene_dir / folder
                dst_path = target_scene_dir / folder
                
                if src_path.exists():
                    if not dst_path.exists():
                        # Only print the "Processing" header the first time we actually do work for this scene
                        if not scene_header_printed:
                            print(f"Processing {scene_name}...")
                            scene_header_printed = True
                        
                        shutil.copytree(src_path, dst_path)
                        set_permissions_recursive(dst_path)
                        print(f"  [OK] Copied and set 777 for {folder}")
                        any_copied = True
                    else:
                        # Commented out to keep the script near-silent on successive runs
                        # print(f"  [SKIP] {folder} already exists")
                        pass
                else:
                    if not scene_header_printed:
                        print(f"Processing {scene_name}...")
                        scene_header_printed = True
                    print(f"  [WARN] {folder} not found in source")

    # Final reporting logic based on whether files were copied
    if not any_copied:
        print("Skipped all because already copied.")
    else:
        print("\nPermissions updated and sync complete!")

if __name__ == "__main__":
    sync_folders()