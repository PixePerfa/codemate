import os
import requests
import re
import functools
import arxiv
import datetime

from rich import print
from parsel import Selector
from typing import List, Literal, Optional
from pydantic import BaseModel
from translate import Translator
from youtube_search import YoutubeSearch
from llama_index.core import Document
from llama_index.readers.github import GithubRepositoryReader, GithubClient


class GithubRepoItem(BaseModel):
    repo_url: str
    desc: str
    lang: str
    name: str
    owner: str
    stars: int
    forks: int


class GithubIssueItem(BaseModel):
    title: str
    labels: List[str]
    created_at: str
    updated_at: str
    closed_at: str
    description: str
    base_commit: str


class GithubDataLoader:
    BASE_TRENDING_URL = "https://github.com/trending"
    BASE_GITHUB_API_URL = "https://api.github.com"

    ENG_TRANSLATOR = Translator(to_lang="en", from_lang="zh")

    LANG_EXTENSIONS_DICT = {
        "python": [".py"],
        "javascript": [".js"],
        "java": [".java"],
        "c": [".c"],
        "cpp": [".cpp", ".cc", ".cxx"],  # Including common C++ file extensions
        "csharp": [".cs"],
        "go": [".go"],
        "ruby": [".rb"],
        "swift": [".swift"],
        "php": [".php"],
        "typescript": [".ts"],
        "kotlin": [".kt"],
        "rust": [".rs"],
        "scala": [".scala"],
        "perl": [".pl"],
        "haskell": [".hs"],
    }

    # folders in a repo to exclude while loading
    INVALID_FOLDERS = ["legacy", ".", "mock"]

    def __init__(self, github_api_key: str) -> None:
        self._github_api_key = github_api_key
        os.environ["GITHUB_TOKEN"] = self._github_api_key

    def mock(self):
        mock_list = [
            "AnswerDotAI/fsdp_qlora",
            "facebookresearch/llama-recipes",
            "lucidrains/diffusion-policy",
            "lucidrains/DALLE2-pytorch",
            "lucidrains/imagen-pytorch",
            "lucidrains/vit-pytorch",
            "lucidrains/denoising-diffusion-pytorch",
        ]
        return self.search_repos_by_name(mock_list)

    def load_repo(
        self,
        owner: str,
        repo: str,
        filter_by_lang: bool = True,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> List[Document]:
        github_client = GithubClient(github_token=self._github_api_key, verbose=True)
        # find the language of the repository and only load the files having extensions from that language.
        repo_item = self.search_repos_by_name(repo_names=[repo])
        assert repo_item
        repo_item = repo_item[0]
        repo_lang = repo_item.lang.lower()
        repo_lang = repo_lang if repo_lang != "c++" else "cpp"
        lang_extensions = self.LANG_EXTENSIONS_DICT.get(repo_lang, [])
        filter_file_extensions = None
        if filter_by_lang:
            assert lang_extensions, f"Not able to find corresponding file extensioons for {repo}: {repo_lang}"
            filter_file_extensions = (
                lang_extensions,
                GithubRepositoryReader.FilterType.INCLUDE,
            )

        if branch is None and commit_sha is None:
            # find the default branch using github api.
            request_url = f"{self.BASE_GITHUB_API_URL}/repos/{owner}/{repo}"

            response = requests.get(request_url, headers={})
            assert response.status_code == 200

            # Get the default branch name
            branch = response.json().get("default_branch")

        return GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_directories=(
                self.INVALID_FOLDERS,
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
            filter_file_extensions=filter_file_extensions,
        ).load_data(branch=branch, commit_sha=commit_sha)

    def search_repos_by_name(
        self, repo_names: List[str], timeout: int = 50, **kwargs
    ) -> List[GithubRepoItem]:
        """
        Returns a list of githubrepoitems belonging to the list of provided repo names.
        """
        if not repo_names:
            return []
        func = getattr(requests, "get")
        headers = kwargs.pop("headers", {})
        items = []

        for name in repo_names:
            url = f"{self.BASE_GITHUB_API_URL}/search/repositories?q={name}"

            response = func(url, headers=headers, timeout=timeout, **kwargs)
            if response.status_code != 200:
                assert f"Unable to retrieve {url}"
            print(url)
            response = response.json()
            for item in response["items"]:
                repo = f'https://github.com/{item["full_name"]}'
                if (
                    item.get("stargazers_count", 0) < 100
                    or not item.get("full_name", "")
                    or not item.get("description", "")
                    or not item.get("language", "")
                ):
                    continue
                desc = "".join(item["description"]).strip()
                name, owner = repo.split("/")[-1], repo.split("/")[-2]
                items.append(
                    GithubRepoItem(
                        repo_url=repo,
                        desc=desc,
                        lang=item["language"],
                        name=name,
                        owner=owner,
                        stars=item["stargazers_count"],
                        forks=item.get("forks_count", 0),
                    )
                )

        return items

    def search_repos_by_topic(
        self,
        topics: List[str],
        timeout: int = 50,
        min_stars: int = 1_000,
        top_k: int = -1,
        **kwargs,
    ) -> List[GithubRepoItem]:
        """
        Returns a list of github repositories which belong to all topics
        provided in the list.
        To reduce noise, we only return repos which have more than min_stars.
        """
        if not topics:
            return []
        headers = kwargs.pop("headers", {})
        func = getattr(requests, "get")
        items = []

        for topic in topics:
            topic = topic.strip()
            topic_q = f"topic:{topic}"
            url = f"{self.BASE_GITHUB_API_URL}/search/repositories?q={topic_q}"

            response = func(url, headers=headers, timeout=timeout, **kwargs)
            if response.status_code != 200:
                assert f"Unable to retrieve {url}"
            response = response.json()
            for item in response["items"]:
                if (
                    item.get("stargazers_count", 0) < min_stars
                    or not item.get("full_name", "")
                    or not item.get("description", "")
                    or not item.get("language", "")
                ):
                    continue
                repo = f'https://github.com/{item["full_name"]}'
                orig_desc = "".join(item["description"]).strip()
                if any(ord(char) > 127 for char in orig_desc):
                    trans_desc = self.ENG_TRANSLATOR.translate(orig_desc)
                else:
                    trans_desc = orig_desc
                name, owner = repo.split("/")[-1], repo.split("/")[-2]
                items.append(
                    GithubRepoItem(
                        repo_url=repo,
                        desc=trans_desc,
                        lang=item["language"],
                        name=name,
                        owner=owner,
                        stars=item["stargazers_count"],
                        forks=item.get("forks_count", 0),
                    )
                )

        items = sorted(items, key=lambda x: x.stars)
        if top_k > 0:
            items = items[:top_k]
        return items

    def get_trending_repos(
        self,
        since: Literal["daily", "weekly", "monthly"] = "monthly",
        timeout: int = 50,
        top_k: int = -1,
        **kwargs,
    ) -> List[GithubRepoItem]:
        """
        Returns a list of trending github repositories filtered by since
        Set since="daily" or since="weekly" to get daily or weekly trending repos.
        """
        headers = kwargs.pop("headers", {})
        func = getattr(requests, "get")
        url = f"{self.BASE_TRENDING_URL}/?since={since}"
        response = func(url, headers=headers, timeout=timeout, **kwargs)
        if response.status_code != 200:
            assert f"Unable to retrieve {url}"
        text = response.text

        li = Selector(text=text).css("[data-hpc]")[0].css("article")
        items = []

        def get_list_num(arr: List[str]):
            return int("".join(re.compile(r"\d+").findall("".join(arr))))

        for article in li:
            repo = article.css("h2")[0].css("::attr(href)").get()
            repo = f"https://github.com/{repo}"
            desc = article.css("p").css("::text").getall()
            desc = "".join(desc).strip()
            if not desc:
                continue
            if any(ord(char) > 127 for char in desc):
                trans_desc = self.ENG_TRANSLATOR.translate(desc)
            else:
                trans_desc = desc

            footer = article.css("div")[2]
            stars = 0
            forks = 0
            for s_or_f in footer.css("div > a"):
                tmp_href = s_or_f.css("::attr(href)").get()
                if tmp_href.endswith("/forks"):
                    forks = get_list_num(s_or_f.css("::text").getall())
                else:
                    stars = get_list_num(s_or_f.css("::text").getall())

            lang_span = footer.css("div > span:has(span)")
            lang = (
                lang_span[0].css("span")[2].css("::text").get()
                if len(lang_span) > 0
                else ""
            )

            build_span = footer.css("div > span:has(a)")
            bb_list = []
            if len(build_span) > 0:
                build_links = build_span[0].css("a")
                for link in build_links:
                    by = link.css("::attr(href)").get()
                    bb_list.append(by)
            name, owner = repo.split("/")[-1], repo.split("/")[-2]
            items.append(
                GithubRepoItem(
                    repo_url=repo,
                    desc=trans_desc,
                    lang=lang,
                    stars=stars,
                    forks=forks,
                    name=name,
                    owner=owner,
                )
            )

        items = sorted(items, key=lambda x: -x.stars)
        if top_k > 0:
            items = items[:top_k]
        return items

    def load_issue(
        self, name: str, owner: str, issue_number: int, **kwargs
    ) -> GithubIssueItem:
        url = f"{self.BASE_GITHUB_API_URL}/repos/{owner}/{name}/issues/{issue_number}"
        func = getattr(requests, "get")
        headers = kwargs.pop("headers", {})

        response = func(url, headers=headers, timeout=60, **kwargs)
        if response.status_code != 200:
            assert f"Unable to retrieve {url}"
        response = response.json()
        labels = response.get("labels", [])
        labels = list(labels) if isinstance(labels, dict) else labels

        # find the base commit closest to created_at.
        commits_url = f"{self.BASE_GITHUB_API_URL}/repos/{owner}/{name}/commits"
        creation_date = response.get("created_at", "")
        assert creation_date, "Invalid creation date"
        creation_date = datetime.datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%SZ")
        since_date = creation_date - datetime.timedelta(days=30)
        until_date = creation_date

        # Get commits around the creation date
        params = {
            "since": since_date.isoformat() + "Z",
            "until": until_date.isoformat() + "Z",
        }
        commit_response = requests.get(commits_url, headers=headers, params=params)
        assert commit_response.status_code == 200
        commit_history = commit_response.json()
        assert commit_history

        return GithubIssueItem(
            title=response.get("title", ""),
            labels=[label.get("name", "") for label in labels],
            created_at=response.get("created_at", ""),
            updated_at=response.get("updated_at", ""),
            closed_at=response.get("closed_at", ""),
            description=response.get("body", ""),
            base_commit=commit_history[0]["sha"],
        )


class ArxivPaperItem(BaseModel):
    published: str
    title: str
    summary: str
    paper_url: str


class ArxivDataLoader:
    def __init__(self):
        self._arxiv_search = functools.partial(
            arxiv.Search,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            id_list=[],
        )

    def mock(self):
        results = []
        for result in self._arxiv_search(
            id_list=[
                "2403.04642",
                "2402.17177v2",
                "2403.03890",
                "2402.18668",
                "2303.04137",
            ]
        ).results():
            results.append(
                ArxivPaperItem(
                    published=str(result.published.date()),
                    title=result.title,
                    summary=result.summary,
                    paper_url=str(result),
                )
            )
        return results

    def load_data(
        self, query: List[str], get_latest: bool = False
    ) -> List[ArxivPaperItem]:
        """
        Loads data from arxiv for papers published relevant to provide topic.
        If get_latest is set to True, only returns papers published within 1 week.
        """
        if not query:
            return []
        items = []
        for query_ in query:
            results = self._arxiv_search(
                query_, max_results=5 if get_latest else 100
            ).results()
            current_date = datetime.datetime.now().date()

            for result in results:
                try:
                    date = datetime.datetime.strptime(
                        str(result.published.date()), "%Y-%m-%d"
                    ).date()
                    diff = (current_date - date).days
                    if get_latest and (diff < 0 or diff > 7):
                        continue
                    items.append(
                        ArxivPaperItem(
                            published=str(result.published.date()),
                            title=result.title,
                            summary=result.summary,
                            paper_url=str(result),
                        )
                    )
                except Exception as e:
                    print("Error while downloading: " + str(e))

        return items


class YoutubeTranscriptItem(BaseModel):
    video_url: str
    num_views: int
    duration: int  # in seconds
    channel: str
    title: str


class YoutubeDataLoader:
    BASE_URL = "https://www.youtube.com"

    def mock(self):
        mock_list = [
            "CUDA Performace Checklist by CUDA MODE",
            "Deep Learning Foundations by Soheil Feizi : Diffusion Models",
            "Stanford Seminar - Robot Skill Acquisition: Policy Representation and Data Generation",
            "Building long context RAG with RAPTOR from scratch",
        ]
        mock_results = []
        for topic in mock_list:
            results = self.search_by_topic(
                [topic], min_views=1, max_results=1, max_duration_secs=2 * 60 * 60
            )
            mock_results.extend(results)
        return mock_results

    def search_by_topic(
        self,
        topics: List[str],
        min_views: int = 1000,
        max_results: int = 100,
        max_duration_secs: int = 1800,
    ) -> List[YoutubeTranscriptItem]:
        items = []
        for topic in topics:
            results = YoutubeSearch(topic, max_results=max_results).to_dict()

            for result in results:
                timelets = result["duration"].strip().split(":")[::-1]
                duration = int(timelets[0])
                if len(timelets) > 1:
                    duration += 60 * int(timelets[1])
                if len(timelets) > 2:
                    duration += 3600 * int(timelets[2])
                if duration >= max_duration_secs:
                    continue

                num_views = int(
                    result["views"].strip().split("views")[0].strip().replace(",", "")
                )
                if num_views < min_views:
                    continue
                url = result["url_suffix"]
                url = f"{self.BASE_URL}{url}"
                items.append(
                    YoutubeTranscriptItem(
                        video_url=url,
                        num_views=num_views,
                        duration=duration,
                        channel=result["channel"],
                        title=result["title"],
                    )
                )

        return items
