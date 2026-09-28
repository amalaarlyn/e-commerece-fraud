import os
import subprocess

def add_all():
    print("Hashing and adding files to git index...")
    for root, dirs, files in os.walk('.'):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root:
            continue
        for file in files:
            path = os.path.join(root, file)
            path = os.path.normpath(path)
            
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                hash_obj_proc = subprocess.run(
                    ['git', 'hash-object', '-w', '--stdin'],
                    input=content,
                    capture_output=True,
                    check=True
                )
                blob_hash = hash_obj_proc.stdout.decode().strip()
                
                mode = oct(os.stat(path).st_mode)[-6:]
                if not mode.startswith('10'):
                    mode = '100644'
                
                subprocess.run(
                    ['git', 'update-index', '--add', '--cacheinfo', f'{mode},{blob_hash},{path}'],
                    check=True
                )
                print(f"Added {path}")
            except Exception as e:
                print(f"Failed to add {path}: {e}")

if __name__ == '__main__':
    add_all()
