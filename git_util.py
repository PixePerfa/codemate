from urllib.parse import urlparse
from github import Github
from git import Repo


def clone_repo(github_url, github_access_token):
    """
      Clone the GitHub repository to a local directory
    """
    g = Github(github_access_token) if github_access_token else Github()
    def extract_repo_name(github_url):
        path = urlparse(github_url).path
        repo_name = path.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        return repo_name


    def extract_repo_details(url):
        path = urlparse(url).path.lstrip('/')
        repo_owner, repo_name = path.split('/')
        repo_name = repo_name.rstrip('.git')
        return repo_owner, repo_name
    repo_owner, repo_name = extract_repo_details(github_url)
    local_dir = f"./{extract_repo_name(github_url)}"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
        print(f"Directory '{local_dir}' deleted.")

    try:
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        default_branch = repo.default_branch
    except Exception as e:
        default_branch = "main"

    try:
        Repo.clone_from(github_url, local_dir)
    except Exception as e:
        print(f"Directory {local_dir} is not empty.")

    return local_dir, default_branch, None