
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================
# CONFIGURATION
# ==========================================

REPO_ID = "arijitsen4321/Tourism_Project_Arijit_Sen"
REPO_TYPE = "dataset"

TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not found.\n"
        "Run:\n"
        "os.environ['HF_TOKEN'] = 'your_huggingface_write_token'"
    )

# ==========================================
# INITIALIZE API
# ==========================================

api = HfApi(token=TOKEN)

# ==========================================
# VERIFY LOGIN
# ==========================================

try:
    user_info = api.whoami()
    print(f"Logged in as: {user_info['name']}")
except Exception as e:
    raise Exception(
        f"Authentication failed.\n"
        f"Check your HF_TOKEN.\n\n{e}"
    )

# ==========================================
# CREATE DATASET IF MISSING
# ==========================================

try:
    api.repo_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )
    print(f"Dataset already exists: {REPO_ID}")

except RepositoryNotFoundError:
    print(f"Dataset not found. Creating: {REPO_ID}")

    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        token=TOKEN,
        exist_ok=True
    )

    print("Dataset created successfully.")

# ==========================================
# UPLOAD DATA
# ==========================================

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

print("Upload completed successfully.")
print(f"https://huggingface.co/datasets/{REPO_ID}")
