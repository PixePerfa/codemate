import os
import re
import json
import instructor
import requests
import globals
import openai

from rich import print
from typing import Dict, Any, List, Optional
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, validator, ValidationError
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from openai import OpenAI as OG_OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.base.response.schema import Response

from llama_index.agent.openai import OpenAIAgent
from llama_index.packs.code_hierarchy import (
    CodeHierarchyNodeParser,
)
from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.core.text_splitter import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine

from external_data_loader import GithubDataLoader, GithubRepoItem
from db_util import DBConfig, VectorDB


class AgentResponse(BaseModel):
    message: Any
    content: Any


def tool_map_to_markdown(tool_map: Dict[str, Any], depth: int, max_depth: int) -> str:
    markdown = ""
    indent = "  " * depth  # Two spaces per depth level
    if depth > max_depth:
        return markdown + f"{indent}-...\n"

    for key, value in tool_map.items():
        if isinstance(value, dict):  # Check if value is a dict
            # Add the key with a bullet point and increase depth for nested dicts
            markdown += (
                f"{indent}- {key}\n{tool_map_to_markdown(value, depth + 1, max_depth)}"
            )
        else:
            # Handle non-dict items if necessary
            markdown += f"{indent}- {key}: {value}\n"

    return markdown


def chunk_repo(github_item: GithubRepoItem, github_access_token):
    repo_lang = github_item.lang

    github_loader = GithubDataLoader(github_api_key=github_access_token)

    print(
        f"Loading repo {github_item.owner}:{github_item.name}:{github_item.commit_sha}"
    )

    documents = github_loader.load_repo(
        owner=github_item.owner,
        repo=github_item.name,
        commit_sha=github_item.commit_sha,
    )
    for doc in documents:
        assert "file_path" in doc.metadata
        doc.metadata["filepath"] = doc.metadata["file_path"]

    nodes = CodeHierarchyNodeParser(
        language=repo_lang,
        # You can further parameterize the CodeSplitter to split the code
        # into "chunks" that match your context window size using
        # chunck_lines and max_chars parameters, here we just use the defaults
        code_splitter=CodeSplitter(language=repo_lang, max_chars=2000, chunk_lines=50),
    ).get_nodes_from_documents(documents)

    return nodes


class GithubSearchTool(BaseToolSpec):
    """
    Uses gitub to search relevant items.
    """

    spec_functions = [
        "find_trending_github_repos",
        "find_relevant_repos",
        "get_github_issue_repo_item",
        "get_github_repo_items",
    ]

    def __init__(self):
        self._github_api_key = os.environ.get("GITHUB_TOKEN", None)
        assert self._github_api_key is not None

    def generate_topics_from_query(self, query: str):
        # Transform embedding to search a DB storing github repo-descriptions.
        class Response(BaseModel):
            output: List[str]

        client = instructor.from_openai(OG_OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o",
            response_model=Response,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert at understanding queries of a software engineer and related it with possible github repositories.
                    Your rules is to understand the query and transform into possible github repository topics.
                    Follow these guidelines:
                    1) Do not create length topics more than 5 words.
                    2) Always use the key term in all topics. For example if query is about LLM always use LLM in each topic.
                    3) Perform reasoning to find possible releated topics even though the terms are not mentioned in the query.
                    3) Do not use any irrelevant terms from the query to make a topic and all topics should be similar in context but
                    different enough to help find all relevant github repos related to the query.
                    4) Do not split key terms to make a new topic. For example Do not do 'LLM finetuning' -> ['LLM', 'finetuning']
                    5) Make as less topics as possible because using these topics to search Github is expensive and rate limited. No more than 5

                    Example:
                    query: "relevant github repositories for LLM finetuning."
                    Correct Answer:
                        topics: "['LLM finetuning', 'language models']"
                    Wrong Answer:
                        topics: "['LLM finetuning', 'LLM', 'finetuning']"
                    query: "How are transformers implemented?"
                    Correct Answer:
                        topics : "['transformers']"

                    """,
                },
                {
                    "role": "user",
                    "content": f"Generate a list of topics for query: {query} which can be used to do github search.",
                },
            ],
        )
        return response.output

    def check_if_repo_relevant(self, query: str, github_item: GithubRepoItem):
        class Response(BaseModel):
            is_relevant: bool
            reason: str

        client = instructor.from_openai(OG_OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o",
            response_model=Response,
            messages=[
                {
                    "role": "system",
                    "content": """

                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                 Given the query and github repo below, check if it is relevant or not:

                 Query:
                 {query}

                 Github Repo:
                    URL: {github_item.repo_url}
                    Descripption: {github_item.desc}
                    Language: {github_item.lang}
                    Name: {github_item.name}
                    Owner: {github_item.owner}
                    Stars: {github_item.stars}
                    Forks: {github_item.forks}

                 """,
                },
            ],
        )
        return response

    def find_trending_github_repos(self, query: Optional[str] = None) -> AgentResponse:
        """Returns a list storing trending GithubRepoItem containing information about the repos.

        Args:
            query: If provided, then filters the trending repos which have description most similar
                to the query, otherwise returns all the trending repos currently.
        """
        items = GithubDataLoader(self._github_api_key).get_trending_repos(
            since="daily", top_k=50
        )
        if query is not None:

            class RelevantRepos(BaseModel):
                repo: GithubRepoItem
                relevance_reason: str

            relevant_items = []
            for item in items:
                response = self.check_if_repo_relevant(query, item)
                if response.is_relevant:
                    relevant_items.append(
                        RelevantRepos(repo=item, relevance_reason=response.reason)
                    )
            return AgentResponse(
                message="List of all trending github repos related to query with reason for relevance.",
                content=relevant_items,
            )

        return AgentResponse(
            message="All trending github repos currently.", content=items
        )

    def find_relevant_repos(self, query: str, top_k: int = 1) -> AgentResponse:
        """Given a query, finds all relevant topics which can be searched on github and returns relevant github repos."""

        topics = self.generate_topics_from_query(query)

        items = GithubDataLoader(self._github_api_key).search_repos_by_topic(
            topics=topics, min_stars=0, top_k=top_k
        )
        return AgentResponse(
            message=f"""
            Relevant repos for {query} based on the following topics generated:
            {topics}
            """,
            content=items,
        )

    def get_github_issue_repo_item(self, issue_url: str) -> AgentResponse:
        """Given a github issue URL, creates a GithubIssueItem which stores all the relevant information
        needed to solve the issue. Also stores the GithubRepoItem storing the commit_sha which needs to
        be used to solve it.

        Example issue_url: https://github.com/sympy/sympy/issues/26134
        """
        regex = re.compile(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)")
        match = regex.match(issue_url)
        if not match:
            return AgentResponse(
                message=f"""
                Invalid issue url sent:{issue_url}
                Check out an example of valid issue URL here: https://github.com/sympy/sympy/issues/26134
                """,
                content=None,
            )
        owner, repo, issue_number = match.groups()
        issue_item = GithubDataLoader(self._github_api_key).load_issue(
            name=repo, owner=owner, issue_number=int(issue_number)
        )
        return AgentResponse(
            message="""
                GithubIssueItem storing the all relevant information needed to solve the issue and GithubRepoItem
                which stores the base_commit which needs to be checked out to solve it.
                """,
            content=issue_item,
        )

    def get_github_repo_items(self, query: str) -> AgentResponse:
        """Given a query, performs query classification to construct the most relevant GithubRepoItems
        and returns as part of response content.

        Following the types of queries supported:

        1) where github URL is explicitly given in query. Uses the githubURL to construct the item.
        2) when github issue URL is explicitly given in query.
        3) for any general query related to particular topic search.
        """

        # query classification
        class Response(BaseModel):
            github_repo_url: List[str] = []
            github_issue_url: List[str] = []
            requires_external_search: bool

            @validator("github_repo_url", pre=True)
            def validate_github_repo_url(cls, value):
                regex = re.compile(r"https://github\.com/([^/]+)/([^/]+)")
                for _val in value:
                    match = regex.match(_val)
                    if not match:
                        raise ValueError(
                            f'repo_url {_val} must start with "https://github.com/"'
                        )
                return value

            @validator("github_issue_url", pre=True)
            def validate_github_issue_url(cls, value):
                regex = re.compile(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)")
                for _val in value:
                    match = regex.match(_val)
                    if not match:
                        raise ValueError(
                            f'issue url {_val} must start with "https://github.com/" and have "issues" and integer issue ID'
                        )
                return value

        client = instructor.from_openai(OG_OpenAI())
        response = client.chat.completions.create(
            model="gpt-4o",
            response_model=Response,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert at understanding github related queries. Sometimes a query
                    directly mentions a github repository or github issue URL. Referring the examples
                    below find out all the relevant URLS mentioned in the query.

                    Also, you need to tell if to solve the query we need to do external search to find
                    relevant github repositories etc.

                    Example queries and expected responses:

                    Query #1: "How is transformer implemented in https://github.com/abc/xyz
                    Response:
                        github_repo_url: ["https://github.com/abc/xyz"]
                        github_issue_url: []
                        requires_github_topic_search: False

                    Query #2: "Find me relevant files to solve issue https://github.com/hello/world/issues/7
                    Response:
                        github_repo_url: []
                        github_issue_url: ["https://github.com/hello/world/issues/7"]
                        requires_github_topic_search: False

                    Query #3: "I want to learn about transformers."
                    Response:
                        github_repo_url: []
                        github_issue_url: []
                        requires_github_topic_search: True

                    Make sure to not mark any URL specified in the query as repo url or issue_url. They should
                    pass the validations specified in the response model.
                    """,
                },
                {
                    "role": "user",
                    "content": f"Help me understand about this query: {query}.",
                },
            ],
        )

        github_repo_items = []

        if response.requires_external_search:
            github_repo_items.extend(self.find_relevant_repos(query).content)
        for repo_url in response.github_repo_url:
            pattern = r"https://github\.com/([^/]+)/([^/]+)"
            match = re.match(pattern, repo_url)
            owner = match.group(1)
            repo = match.group(2)
            full_name = f"{owner}/{repo}"
            github_repo_items.append(
                GithubDataLoader(self._github_api_key).search_repos_by_name(
                    repo_names=[full_name]
                )
            )
        for issue_url in response.github_issue_url:
            issue_content = self.get_github_issue_repo_item(issue_url).content
            if issue_content:
                github_repo_items.append(issue_content.repo_item)
        if not github_repo_items:
            return AgentResponse(
                message=f"Not able to find any github repo items for query: {query}. Rewrite the query and try again.",
                content=github_repo_items,
            )

        return AgentResponse(
            message=f"""
                Found list of repo items relevant for query: {query}
                """,
            content=github_repo_items,
        )


class InternalDatabaseSearch(BaseToolSpec):
    """
    Uses a maintained vector db to search for relevant items given query.
    """

    spec_functions = ["retrieve_relevant_repos", "add_github_repos"]

    def __init__(
        self,
        metadata_db_config: DBConfig,
        code_db_config: DBConfig,
    ):
        self.client = instructor.from_openai(OG_OpenAI())

        self.metadata_db = VectorDB(metadata_db_config)
        self.code_db = VectorDB(code_db_config)
        self.github_access_token = os.environ.get("GITHUB_TOKEN", None)
        assert self.github_access_token is not None

    def retrieve_relevant_repos(self, query: str, top_k: int = 5) -> AgentResponse:
        """Retrieve top k relevant code repos given the query."""

        # Transform embedding to search a DB storing github repo-descriptions.
        class Questions(BaseModel):
            questions: List[str]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=Questions,
            messages=[
                {
                    "role": "system",
                    "content": """
                 You are an expert at understanding queries of a software engineer and comparing with description of popular
                 github repositories to perform retrieval.
                 Your rules is to understand the query and transform into possible github repsository descriptions.

                 Follow these rules/guidelines:
                 1) Do not create very lengthy description. It should be more than 500 words.
                 2) Find the key term which in the query and always use it while making the description.
                    For example, if query is about LLM always use LLM in each topic.
                 3) Do not use any irrelevant terms from the query to make a description.
                 4) Each description should be different enough to do all possible similarity search to get the best retrieval.
                 5) Do not split key terms to make a new description context. For example Do not do 'LLM finetuning' -> ['LLM', 'finetuning']
                 """,
                },
                {
                    "role": "user",
                    "content": f"Generate 4 descriptions for query: {query}",
                },
            ],
        )
        questions = response.questions
        response = self.metadata_db.search(query_texts=questions, top_k=top_k)
        if not len(response["metadatas"]):
            return AgentResponse(
                message="""
                No relevant code repos found for query: {query} based on retrieval from following generated descriptions:

                {questions}
                """,
                content=None,
            )

        class RelevantRepoItem(BaseModel):
            summary: str
            repo_url: str
            desc: str
            lang: str
            name: str
            owner: str
            stars: int
            forks: int

        relevant_items = []
        for i in range(len(response["metadatas"])):
            print(response["metadatas"][i])
            relevant_items.append(
                RelevantRepoItem(
                    summary=response["documents"][i][0],
                    repo_url=response["metadatas"][i][0]["repo_url"],
                    desc=response["metadatas"][i][0]["desc"],
                    lang=response["metadatas"][i][0]["lang"],
                    name=response["metadatas"][i][0]["name"],
                    owner=response["metadatas"][i][0]["owner"],
                    stars=response["metadatas"][i][0]["stars"],
                    forks=response["metadatas"][i][0]["forks"],
                )
            )

        return AgentResponse(
            message="""
            Found code repos for query: {query} based on following generated descriptions:

            {questions}
            """,
            content=relevant_items,
        )

    def generate_repo_summary(self, prompt: str) -> str:
        class SummaryResponse(BaseModel):
            github_url: str
            summary: str

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=SummaryResponse,
            messages=[
                {
                    "role": "system",
                    "content": """
                 You are principle engineer and you are summarizing a codebase to provide users with a good summary of the code.
                 Your job is to make sure the summary should have high retrieval accurary for tasks like finding relevant repos
                 given a user query.

                 Follow these guidelines:
                 1) No bullet points or list.
                 2) Do not mention any URLs or direct reference of the repository in the summary.
                 3) Do not mention the name of functions, modules and folder names directly from repository structure in the summary.
                 4) Try to figure out possible code given the organization of code and function names and come up with
                 appropriate description for the entire code repository without specifying the names of modules and functions.
                 5) Wherever ... is mentioned in the repository structure, assume there are more folders and files not mentioned there.
                 """,
                },
                {"role": "user", "content": f"""{prompt}"""},
            ],
        )
        return response.summary

    def add_github_repos(self, repo_items: List[GithubRepoItem]) -> AgentResponse:
        """Upserts the internal database if any github repo item is absent."""
        added_items = []
        present_items = []

        for item in repo_items:
            filter_query = {"repo_url": item.repo_url}
            if item.commit_sha:
                filter_query.update({"commit_sha": item.commit_sha})

            metadata_nodes = self.metadata_db.filter(filter_query)
            code_nodes = self.code_db.filter(filter_query)

            if len(metadata_nodes["metadatas"]) and len(code_nodes["metadatas"]):
                del code_nodes
                for i in range(len(metadata_nodes["metadatas"])):
                    present_items.append(metadata_nodes["metadatas"][i])
                    present_items[-1].update(
                        {"summary": metadata_nodes["documents"][i]}
                    )
            else:
                nodes = chunk_repo(item, self.github_access_token)
                (
                    tool_map_dict,
                    markdown_dict,
                ) = CodeHierarchyNodeParser.get_code_hierarchy_from_nodes(
                    nodes, max_depth=0
                )
                print(markdown_dict)
                num_tries = 3
                max_depth_to_consider = 3
                summary = None
                while num_tries:
                    tool_map = tool_map_to_markdown(
                        tool_map_dict, 0, max_depth=max_depth_to_consider
                    )

                    summary_prompt = f"""

                    Repository Details:

                    URL: {item.repo_url}
                    Descripption: {item.desc}
                    Language: {item.lang}
                    Name: {item.name}
                    Owner: {item.owner}
                    Stars: {item.stars}
                    Forks: {item.forks}

                    Here is how the repository structure looks like with folder, module names present etc.

                    {tool_map}
                    """

                    try:
                        summary = self.generate_repo_summary(summary_prompt)
                        break
                    except Exception:
                        max_depth_to_consider -= 1
                    num_tries -= 1
                assert summary is not None

                # add commit to the ID to have separate snapshots of codebase if needed.
                id_prefix = (
                    item.repo_url + item.commit_sha
                    if item.commit_sha
                    else item.repo_url
                )

                self.metadata_db.add(
                    documents=[summary],
                    metadatas=[item.dict()],
                    ids=[id_prefix],
                )
                # filter out code documents which can't be embedding.
                documents = [node.text for node in nodes if node.text]
                metadatas = [node.dict() for node in nodes if node.text]

                for m in metadatas:
                    cur_keys = list(m.keys())
                    for k in cur_keys:
                        if m[k] is None:
                            m.pop(k)
                        elif isinstance(m[k], dict):
                            m[k] = json.dumps(m[k])
                        elif isinstance(m[k], list):
                            m[k] = json.dumps(m[k])

                self.code_db.add(
                    documents=documents,
                    ids=list(map(lambda x: f"{id_prefix}_{x}", range(len(documents)))),
                    metadatas=list(
                        map(
                            lambda x: {
                                **item.dict(),
                                "node_id": x,
                                **metadatas[x],
                            },
                            range(len(documents)),
                        )
                    ),
                )
                added_items.append(item.dict())
                added_items[-1].update({"summary": summary})
        if added_items:
            return AgentResponse(
                message="New github URLs are now added to the internal database. Please try using them again",
                content=added_items,
            )
        return AgentResponse(
            message="All github URLs are already present in the database. They are ready to use",
            content=present_items,
        )


class CodeAnalyzerTool(BaseToolSpec):
    """
    Loads github repositories and answer any question for a given code repository,
    compares the respository.
    """

    spec_functions = [
        "chat",
        "compare",
    ]

    def __init__(self, db_config: DBConfig):
        self._agents = {}
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        assert self._openai_api_key is not None
        openai.api_key = self._openai_api_key
        self.llm = OpenAI(
            temperature=0,
            model_name=globals.GPT_MODEL_NAME,
            api_key=self._openai_api_key,
        )
        self.code_db = VectorDB(db_config)

    @classmethod
    def get_repo_desc(cls, repo_url):
        github_url_pattern = r"https://github\.com/([^/]+/[^/]+)"
        import re

        match = re.search(github_url_pattern, repo_url)
        name = match.group(1)
        name = name.replace(".git", "")
        url = f"https://api.github.com/search/repositories?q={name}"
        func = getattr(requests, "get")
        response = func(url, headers={}, timeout=50)
        if response.status_code != 200:
            assert f"Unable to retrieve {url}"
        response = response.json()
        desc = None
        for item in response["items"]:
            if (
                item.get("stargazers_count", 0) < 100
                or not item.get("full_name", "")
                or not item.get("description", "")
                or not item.get("language", "")
            ):
                continue
            desc = "".join(item["description"]).strip()
            break
        return desc if desc is not None else ""

    def make_code_hierarchy_nodes(self, metadatas):
        nodes = []

        def get_node_relationship(value):
            for member in NodeRelationship:
                if member.value == value:
                    return member
            return None

        for meta in metadatas:
            rels = {
                get_node_relationship(k): v
                for k, v in json.loads(meta["relationships"]).items()
                if v is not None
            }
            nodes.append(
                TextNode(
                    text=meta["text"],
                    id_=meta["id_"],
                    relationships=rels,
                    class_name=meta["class_name"],
                    excluded_embed_metadata_keys=json.loads(
                        meta["excluded_embed_metadata_keys"]
                    ),
                    excluded_llm_metadata_keys=json.loads(
                        meta["excluded_llm_metadata_keys"]
                    ),
                    metadata=json.loads(meta["metadata"]),
                    metadata_seperator=meta["metadata_seperator"],
                    node_id=meta["node_id"],
                    text_template=meta["text_template"],
                    metadata_template=meta["metadata_template"],
                )
            )
        return nodes

    def make_agent(self, repoItems: List[GithubRepoItem]) -> AgentResponse:
        """
        Given a list of github repo urls, create a list of openAI agents to chat with.
        """

        def _make_agent(repoItem, nodes):
            query_engine = CodeHierarchyKeywordQueryEngine(
                nodes=nodes,
            )
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name="code_lookup",
                description=f"Useful for looking up information about the {repoItem.repo_url} codebase.",
            )
            tool_instructions = query_engine.get_tool_instructions()
            system_prompt = f"""

            You are principal software engineer with deep understanding of the following gitub
            repo:
            repo_url: {repoItem.repo_url}
            repo_description: {self.get_repo_desc(repoItem.repo_url)}

            You are onboarding an early AI engineer to understand the the repo's codebase.
            They will give you the query for a repo. You have to access to a code lookup tool
            for the github repo and here are the instructions to use it:

            {tool_instructions}

            Always mention the most relevant pieces of code in your response.

            """
            return OpenAIAgent.from_tools(
                [tool], llm=self.llm, system_prompt=system_prompt, verbose=False
            )

        for repoItem in repoItems:
            repoItemstr = json.dumps(repoItem.dict())
            if repoItemstr in self._agents:
                continue

            filter_query = {"repo_url": repoItem.repo_url}
            if repoItem.commit_sha:
                filter_query.update({"commit_sha": repoItem.commit_sha})

            nodes = self.code_db.filter(filter_query)

            if not len(nodes["metadatas"]):
                return AgentResponse(
                    message="""
                Following github repoItem is not present in the database. Please index them into the internal database first using
                add_github_repos tool from InternalDatabaseSearch Agent.
                """,
                    content=repoItem,
                )

            nodes = self.make_code_hierarchy_nodes(nodes["metadatas"])

            if not len(nodes):
                return AgentResponse(
                    message=f"""
                Not able to process data for {repoItemstr} from internal database. Gently respond the user that we are not able
                to process this repo url to provide any information related to it.
                """,
                    content=repoItem,
                )
            self._agents[repoItemstr] = _make_agent(repoItem, nodes)

        return AgentResponse(
            message="Created agents for all github urls. You can now ask any queries related to them.",
            content=None,
        )

    def _validate_input(self, ip) -> AgentResponse:
        try:
            return AgentResponse(message="", content=GithubRepoItem.parse_obj(ip))
        except ValidationError:
            return AgentResponse(
                message=f"""
                Not able to use input: {ip}.
                Make sure to follow the schema of GithubRepoItem pydantic BaseModel for the input:
                {GithubRepoItem.schema_json()}
                """,
                content=None,
            )

    def chat(self, repoItem: GithubRepoItem, query: str) -> AgentResponse:
        """
        Given a repo item answer the query by chatting with an agent for that repo.

        repoItem: GitubRepoItem Pydantic BaseModel object
        query: Input query for which the chat will be performed on the repoItem.
        """

        validateResponse = self._validate_input(repoItem)
        if validateResponse.message:
            return validateResponse
        else:
            repoItem = validateResponse.content

        agent_response = self.make_agent([repoItem])
        if agent_response.content is not None:
            return agent_response

        repoItemstr = json.dumps(repoItem.dict())
        response = self._agents[repoItemstr].chat(query)

        # get code chunks from both tool based and semantic similarity based

        code_chunks = ""

        for source in response.sources:
            if (
                isinstance(source.raw_output, Response)
                and source.tool_name == "code_lookup"
                and source.raw_output.response
            ):
                code_chunks += source.raw_output.response
                code_chunks += "\n\n"
        response_str = str(response)

        repo_desc = self.get_repo_desc(repoItem.repo_url)
        messages = []
        messages.append(
            ChatMessage(
                role="assistant",
                content="""
        You are principal software engineer with deep understanding of the
        provided github repo. you are onboarding an early AI engineer to understand the
        complexities that repo's codebase. They will give
        you the URL, description and some relevant code from that repo. Think step by step
        using the following guidelines and give a detailed analysis to help onboard. Mention
        relevant code while answering if needed to explain better but make sure to not fill
        the response only with code. try to give explainations.

        If you see 'Code replaced for brevity' then a uuid, in the code-chunks, that means
        the code is cut-out there, so make sure to not mention the UUID in the response.
        """,
            )
        )
        messages.append(
            ChatMessage(
                role="user",
                content=f"""
        Hi, Can you help understand code related to my query from a gitub repo.
        I am listing the URL, github repo description, output from a code lookup tool:


        Query: {query}

        URL: {repoItem.repo_url}
        Description: {repo_desc}
        Code Lookup Tool Output: {response_str}
        """,
            )
        )
        return AgentResponse(message=response.sources, content=str(response))

    def compare(
        self, repoItemA: GithubRepoItem, repoItemB: GithubRepoItem, query: str
    ) -> AgentResponse:
        """
        Given a two github repoItems, compares the code in two repos for a query and yield a response

        repoItemA: GitubRepoItem Pydantic BaseModel object
        repoItemB: GitubRepoItem Pydantic BaseModel object
        query: Input query for which the comparison will be performed on repoItemA and repoItemB.
        """
        # seems like there are no validations done by the agent.
        validateResponseA = self._validate_input(repoItemA)
        validateResponseB = self._validate_input(repoItemB)
        if validateResponseA.message:
            return validateResponseA
        else:
            repoItemA = validateResponseA.content
        if validateResponseB.message:
            return validateResponseB
        else:
            repoItemB = validateResponseB.content

        if repoItemA.repo_url == repoItemB.repo_url:
            return AgentResponse(
                message="Please ask user to pass valid inputs for comparison. Both github repo URLs are identical and can't do comparison on identical things.",
                content=None,
            )
        agent_response = self.make_agent([repoItemA, repoItemB])
        if agent_response.content is not None:
            return agent_response

        repoItemAstr = json.dumps(repoItemA.dict())
        repoItemBstr = json.dumps(repoItemB.dict())
        response_a = str(self._agents[repoItemAstr].chat(query))
        response_b = str(self._agents[repoItemBstr].chat(query))
        # compare the response from both repos and construct the output.
        repo_desc_a = self.get_repo_desc(repoItemA.repo_url)
        repo_desc_b = self.get_repo_desc(repoItemB.repo_url)

        messages = []
        messages.append(
            ChatMessage(
                role="assistant",
                content="""
        You are principal software engineer with deep understanding of python and
        machine learning. you are onboarding an early AI engineer to understand the
        complexities and difference between two github codebases. They will give
        you the URL, description and some code from each repo. Think step by step
        using the following guidelines and give a detailed analysis to help onboard:
        1) Explain each repo's provided code and what is required to use it.
        2) Are both code-chunks doing the same thing?
        3) How is one repo's code better than other?
        """,
            )
        )
        messages.append(
            ChatMessage(
                role="user",
                content=f"""
        Here are the details of two repos, help me compare them and analyze both
        based on my query. I am listing the URL, description and output from a
        code lookup tool related to my query: {query}

        Repo 1:

        URL: {repoItemA.repo_url}
        Description: {repo_desc_a}
        Code Lookup Tool Output: {response_a}


        Repo 2:

        URL: {repoItemB.repo_url}
        Description: {repo_desc_b}
        Code Lookup Tool Output: {response_b}
        """,
            )
        )
        return AgentResponse(
            message=f"""
            Chat response for query: {query} based on relevant code chunks found from both code repos:
            {response_a}

            {response_b}
            """,
            content=self.llm.chat(messages).message.content,
        )


def create_agent(code_db_config: DBConfig, metadata_db_config: DBConfig):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    assert OPENAI_API_KEY is not None
    openai.api_key = OPENAI_API_KEY
    llm = OpenAI(temperature=0, model=globals.GPT_MODEL_NAME)

    github_tool_spec = GithubSearchTool()
    code_analyzer_tool_spec = CodeAnalyzerTool(db_config=code_db_config)

    load_and_search_tool_spec = InternalDatabaseSearch(
        metadata_db_config=metadata_db_config, code_db_config=code_db_config
    )
    code_analyzer_agent = OpenAIAgent.from_tools(
        code_analyzer_tool_spec.to_tool_list(),
        system_prompt="""
        You are helping onboard an early AI engineer.
        Given a query understand if the user is asking question about a specific repository
        or wants to compare two github repositories.
        Follow these guidelines:
        1) Understand the query and give as an input to the relevant tool for chatting or comparison.
        2) Do not make random github URLs yourself for chatting and comparison.
        If user has not provided any URL with their query, ask the user to provide
        the github repo URL they are interested to learn about.
        3) For comparing github repos, always make sure that user intent has two distinct
        github URLs and a query for comparison.
        """,
        llm=llm,
        verbose=True,
        callback_manager=None,
        max_function_calls=100,
    )
    github_search_agent = OpenAIAgent.from_tools(
        github_tool_spec.to_tool_list(),
        system_prompt="""
        Github is world's largest platform for storing, tracking and collaborating
        on various types of software projects. Understand the context, to figure
        out which tool can answer the query best.
        Note: Searching topics on github works well if the topics are concise and
        are at max 2 words. Depending on the context, break into multiple small topics.
        """,
        llm=llm,
        verbose=True,
        max_function_calls=100,
    )
    load_and_search_agent = OpenAIAgent.from_tools(
        load_and_search_tool_spec.to_tool_list(),
        system_prompt="""
        Depending on the query, find any relevant github repos inside the database or ingest
        the repos for further use. Searching external is costly, and we want to use this
        internal database as much as possible to be more efficient.
        """,
        llm=llm,
        verbose=True,
        max_function_calls=100,
    )
    code_analyzer_query_agent = QueryEngineTool.from_defaults(
        code_analyzer_agent,
        name="code_analyzer_query_agent",
        description="""
        Given a  1) question/query  2) github repo URLs,
        Answers or compares using relevant context from gitub repository codebase
        of provide github URLs for given a question/query.
        """,
    )
    github_search_query_agent = QueryEngineTool.from_defaults(
        github_search_agent,
        name="github_search_query_agent",
        description="""
        Use this to search github for relevant or trending repos.
        """,
    )
    load_and_search_query_agent = QueryEngineTool.from_defaults(
        load_and_search_agent,
        name="load_and_search_agent",
        description="""
        Interacts with an internal databases which contains two collections:
        1) Metadata_DB => Stores github repo and their corresponding summary describing about the repository.
        2) Nodes_DB -> Stores all the python code chunked as documents for a given github repo.

        Metadata_DB can be used for quick internal search to find relevant github repos already ingested. Prefer
        this over search externally e.g. doing github API search.

        Nodes_DB should ingest all the repos before answering any questions related to the repos and doing code
        comparisons etc.

        This database should be updated based on interaction so that it becomes more useful as we keep using it
        over time.
        """,
    )

    return OpenAIAgent.from_tools(
        [
            code_analyzer_query_agent,
            github_search_query_agent,
            load_and_search_query_agent,
        ],
        llm=llm,
        verbose=True,
        system_prompt="""
        You are a Software Engineering Mentor helping for codebase onboarding,
        feature design and conducting research by analyzing external code repositories.

        Here is a list of agents you have access to:

        1) code_analyzer_query_agent:
            a) Answers query for a given github repo.
            b) Compare two github repos given a query.

        2) github_search_query_agent:
            a) Trending repos on github
            b) search github API to find relevant repos

        3) load_and_search_agent:
            a) retrieves relevant repos stored in an internal DB.
            b) ingests github repos in the internal DB.

        Take a deep breathe and understand the query to think about how to best use the agents sequentially
        to solve the task and then execute the agents sequentially to solve it.



        Follow the guidelines:
        1) If github repo is not directly mentioned in the query as URL, do external github search and confirm the results with user before proceeding.
        2) If the user is asking directly to load into internal DB, just do that and return the result of loading.
        3) If the query is related to a github repo but it is not evident, check with user if they want to provide the github repo or search externally on github.
        4) Think carefully and understand what query the user is asked about a particular github repo or trying to ask for comparison.
        """,
        max_function_calls=100,
    )
