from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path="./datasets", repo_id="willFowler/Format_ACR", repo_type="dataset")
