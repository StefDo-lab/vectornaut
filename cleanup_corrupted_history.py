import os
import shutil

def cleanup():
    # Directories to clear
    dirs_to_clear = ["history", "reports", "generated_scripts", "generated_tests"]
    
    for folder in dirs_to_clear:
        if os.path.exists(folder):
            print(f"Clearing contents of '{folder}'...")
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"  Deleted: {filename}")
                except Exception as e:
                    print(f"  Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory '{folder}' does not exist.")

if __name__ == "__main__":
    cleanup()
