

import os
import base64
import requests

# -----------------------------
# CONFIGURATION
# -----------------------------
owner = "krob3000"  # Your GitHub username
repo = "Customer-Churn-Example"  # Desired repo name
branch = "main"

# Get GitHub token from environment variable
token = os.getenv("GITHUB_TOKEN")
if not token:
    raise ValueError("❌ GITHUB_TOKEN environment variable not set!")

api_base = "https://api.github.com"
headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

# -----------------------------
# STEP 1: Ensure requirements.txt exists
# -----------------------------
if not os.path.exists("requirements.txt"):
    print("⚠ requirements.txt not found. Generating...")
    os.system("pip freeze > requirements.txt")
    print("✅ requirements.txt created successfully")

files_to_upload = ["README.md", "requirements.txt", "app.py"]

# -----------------------------
# STEP 2: Check if repo exists, create if missing
# -----------------------------
repo_url = f"{api_base}/repos/{owner}/{repo}"
resp = requests.get(repo_url, headers=headers)

if resp.status_code == 404:
    print(f"⚠ Repo '{repo}' not found. Creating...")
    create_resp = requests.post(
        f"{api_base}/user/repos",
        headers=headers,
        json={"name": repo, "private": False, "auto_init": True}
    )
    if create_resp.status_code != 201:
        raise Exception(f"❌ Failed to create repo: {create_resp.text}")
    print(f"✅ Repo '{repo}' created successfully!")
elif resp.status_code == 200:
    print(f"✅ Repo '{repo}' already exists.")
else:
    raise Exception(f"❌ Error checking repo: {resp.text}")

# -----------------------------
# STEP 3: Get branch reference
# -----------------------------
ref_url = f"{api_base}/repos/{owner}/{repo}/git/ref/heads/{branch}"
ref_resp = requests.get(ref_url, headers=headers)

if ref_resp.status_code == 404:
    # Branch doesn't exist, create it from default branch
    default_branch = "main"
    print(f"⚠ Branch '{branch}' not found. Creating...")
    # Get default branch commit SHA
    default_ref_url = f"{api_base}/repos/{owner}/{repo}/git/ref/heads/{default_branch}"
    default_ref_resp = requests.get(default_ref_url, headers=headers)
    if default_ref_resp.status_code != 200:
        raise Exception(f"❌ Failed to get default branch: {default_ref_resp.text}")
    sha = default_ref_resp.json()["object"]["sha"]
    # Create new branch
    create_branch_resp = requests.post(
        f"{api_base}/repos/{owner}/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch}", "sha": sha}
    )
    if create_branch_resp.status_code != 201:
        raise Exception(f"❌ Failed to create branch: {create_branch_resp.text}")
    print(f"✅ Branch '{branch}' created successfully!")
    latest_commit_sha = sha
else:
    latest_commit_sha = ref_resp.json()["object"]["sha"]

# -----------------------------
# STEP 4: Get base tree SHA
# -----------------------------
commit_url = f"{api_base}/repos/{owner}/{repo}/git/commits/{latest_commit_sha}"
commit_resp = requests.get(commit_url, headers=headers)
base_tree_sha = commit_resp.json()["tree"]["sha"]

# -----------------------------
# STEP 5: Create blobs for files
# -----------------------------
blobs = []
for file_path in files_to_upload:
