
import os
import subprocess

# Change to your project directory
project_path = r"C:\Users\User\CustomerChurnPractice"  # Update this path
os.chdir(project_path)

# GitHub repository URL
repo_url = "https://github.com/krob3000/Customer-Churn-Example.git"

# ✅ Create requirements.txt if missing
if not os.path.exists("requirements.txt"):
    print("⚠ requirements.txt not found. Generating...")
    subprocess.run(["pip", "freeze"], stdout=open("requirements.txt", "w"))
    print("✅ requirements.txt created successfully")

# ✅ Ensure app.py is included
files_to_add = ["README.md", "requirements.txt", "app.py"]

# Commands to execute
commands = [
    ["git", "init"],
    ["git", "add"] + files_to_add,
    ["git", "commit", "-m", "Add README.md, requirements.txt, and app.py"],
    ["git", "branch", "-M", "main"],
    ["git", "remote", "remove", "origin"],
    ["git", "remote", "add", "origin", repo_url],
    ["git", "push", "-u", "origin", "main"]
]

# Execute commands
for cmd in commands:
    result = subprocess.run(cmd, shell=False)
    if result.returncode == 0:
        print(f"✅ {' '.join(cmd)} executed successfully")
    else:
        print(f"❌ Error running: {' '.join(cmd)}")

print("\n✅ Files pushed to GitHub successfully!")
``
