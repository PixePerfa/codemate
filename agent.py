import openai
import sys
import os
import chromadb
import shutil
import re
import json
import instructor
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from langsmith import traceable
from llama_index.core.llms import ChatMessage
from datetime import datetime
from pydantic.main import BaseModel
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from openai import OpenAI as OG_OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.base.response.schema import Response

from llama_index.agent.openai import OpenAIAgent
from llama_index.packs.deeplake_deepmemory_retriever import DeepMemoryRetrieverPack
from langchain_community.vectorstores import DeepLake
from llama_index.core.schema import Document
from llama_index.tools.tavily_research.base import TavilyToolSpec
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List, Literal, Optional, Tuple
from github import Github
from git import Repo
from urllib.parse import urlparse
from pathlib import Path
from llama_index.packs.code_hierarchy import (
    CodeHierarchyNodeParser,
)
from llama_index.packs.code_hierarchy import CodeHierarchyKeywordQueryEngine

from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
    CBEvent
)
from llama_index.core.callbacks.schema import (
    BASE_TRACE_EVENT,
    TIMESTAMP_FORMAT,
    CBEvent,
    CBEventType,
    EventStats,
    EventPayload
)
from collections import defaultdict
import logging

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from external_data_loader import GithubDataLoader, GithubRepoItem
from db_util import DBConfig, VectorDB


INVALID_EXTENSIONS = [
    ".xcconfig",
    "macos",
    ".xsl",
    ".csv",
    ".cmake",
    ".h",
    ".cc",
    ".lock",
    ".html",
    ".txt",
    ".pyi",
    ".git",
    ".gitignore",
    ".swp",
    ".swo",
    ".pyc",
    ".pyo",
    ".class",
    ".css",
    ".rst",
    ".dll",
    ".exe",
    ".jar",
    ".o",
    ".a",
    ".min.js",
    ".min.css",
    ".egg",
    ".war",
    ".png",
    ".tar.gz",
    ".zip",
    ".min.js.map",
    ".min.css.map",
    ".bundle.js",
    ".bundle.css",
    ".svelte",
    ".jsx",
    ".vue",
    ".scss",
    ".less",
    ".sass",
    ".min.html",
    ".map",
    ".svg",
    ".woff",
    ".woff2",
    ".class",
    ".jar",
    ".war",
    ".ear",
    ".jad",
    ".pyc",
    ".pyo",
    ".pyd",
    ".egg-info",
    ".dist-info",
    ".pyz",
    ".mod",
    ".sum",
    ".a",
    ".jpeg",
    ".json",
    ".pptx",
    ".xml",
    ".jpg",
    ".tiff",
    ".pdf",
    ".eml",
    ".toml",
    ".dockerignore",
    ".txt",
    ".slug",
    ".apk",
    ".dex",
    ".so",
    ".aar",
    ".class",
    ".jar",
    ".war",
    ".ear",
    ".iml",
    ".idea",
    ".gradle",
    ".buildcache",
    ".externalNativeBuild",
    ".cxx",
    ".kts",
    ".kotlin_module",
    ".properties",
    ".log",
    ".npy",
    ".tmp",
    ".bak",
    ".app",
    ".ipa",
    ".dSYM",
    ".xcworkspace",
    ".xcuserstate",
    ".xccheckout",
    ".xcscmblueprint",
    ".xcuserdatad",
    ".pbxproj",
    ".mode1v3",
    ".mode2v3",
    ".perspectivev3",
    ".xcuserstate",
    ".xcuserdatad",
    ".xcscmblueprint",
    ".xccheckout",
    ".xcodeproj",
    ".xcassets",
    ".storyboard",
    ".strings",
    ".plist",
    ".entitlements",
    ".mobileprovision",
    ".lockfile",
    ".xcuserstate",
    ".xcuserdata",
    ".log",
    ".tmp",
    ".bak",
    ".xib",
    ".pbxproj",
    ".JPG",
    ".poetry",
    ".h5",
    ".ipynb",
    ".avi",
    ".mp4",
    ".pkl",
    ".npy",
    ".bin",
    "BUILD",
    "WORKSPACE",
    "BUILD.bazel",
    "Makefile",
    ".index",
    ".PNG",
    ".png",
    ".msg",
    ".ipynb_checkpoints",
    ".gitignore",
    ".DS_Store",
    ".gif"
]



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
    "haskell": [".hs"]
}





def tool_map_to_markdown(tool_map: Dict[str, Any], depth: int, max_depth: int) -> str:
    markdown = ""
    indent = "  " * depth  # Two spaces per depth level
    if depth > max_depth:
        return markdown + f"{indent}-...\n"

    for key, value in tool_map.items():
        if isinstance(value, dict):  # Check if value is a dict
            # Add the key with a bullet point and increase depth for nested dicts
            markdown += f"{indent}- {key}\n{tool_map_to_markdown(value, depth + 1, max_depth)}"
        else:
            # Handle non-dict items if necessary
            markdown += f"{indent}- {key}: {value}\n"

    return markdown

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



def chunk_repo(github_item: GithubRepoItem, github_access_token):
    repo_url = github_item.repo_url
    repo_lang = github_item.lang
    assert repo_lang
    repo_lang = repo_lang.lower()
    repo_lang = repo_lang if repo_lang != "c++" else "cpp"
    lang_extensions = LANG_EXTENSIONS_DICT.get(repo_lang, [])
    if not lang_extensions:
        return None, None
    try:
        local_dir, _, _ = clone_repo(repo_url, github_access_token)
    except Exception as e:
        return []
    valid_files = []
    for dirpath, _, filenames in os.walk(local_dir):
        for file in filenames:
            _, file_extension = os.path.splitext(file)
            if file_extension.lower() not in lang_extensions:
                continue
            full_path = os.path.join(dirpath, file)
            # ignore folders belonging to tests, legacy
            if 'test' in full_path or 'legacy' in full_path or '.github' in full_path or 'mock' in full_path:
                continue
            valid_files.append(Path(full_path))
    
    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_metadata=lambda x: {"filepath": x},
    ).load_data()

    nodes = CodeHierarchyNodeParser(
        language=repo_lang,
        # You can further parameterize the CodeSplitter to split the code
        # into "chunks" that match your context window size using
        # chunck_lines and max_chars parameters, here we just use the defaults
        # code_splitter=CodeSplitter(language="python"),
    ).get_nodes_from_documents(documents)
    shutil.rmtree(local_dir)
    return nodes, documents


class AgentResponse(BaseModel):
    message: Any
    content: Any

class PythonicallyPrintingBaseHandler(BaseCallbackHandler):
    """
    Callback handler that prints logs in a Pythonic way. That is, not using `print` at all; use the logger instead.
    See https://stackoverflow.com/a/6918596/1147061 for why you should prefer using a logger over `print`.

    This class is meant to be subclassed, not used directly.

    Using this class, your LlamaIndex Callback Handlers can now make use of vanilla Python logging handlers now.
    One popular choice is https://rich.readthedocs.io/en/stable/logging.html#logging-handler.
    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger: Optional[logging.Logger] = logger
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def _print(self, str) -> None:
        if self.logger:
            self.logger.debug(str)
        else:
            # This branch is to preserve existing behavior.
            print(str, flush=True)

class CustomCallbackHandler(PythonicallyPrintingBaseHandler):
    """Callback handler that keeps track of debug info.

    NOTE: this is a beta feature. The usage within our codebase, and the interface
    may change.

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of and debug events.

    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.

    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        print_trace_on_end: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the llama debug handler."""
        self._event_pairs_by_type: Dict[CBEventType, List[CBEvent]] = defaultdict(list)
        self._event_pairs_by_id: Dict[str, List[CBEvent]] = defaultdict(list)
        self._sequential_events: List[CBEvent] = []
        self._cur_trace_id: Optional[str] = None
        self._trace_map: Dict[str, List[str]] = defaultdict(list)
        self.print_trace_on_end = print_trace_on_end
        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
            logger=logger,
        )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.
            parent_id (str): parent event id.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        for payload_type in event.payload:
            if payload_type == EventPayload.TOOL:
                debug_tool = (event.payload[payload_type].to_openai_tool())
                if debug_tool["function"]["name"] == "load":
                    print(f"Loading github repos....")
                elif debug_tool["function"]["name"] == "compare":
                    print("Comparing github repos!!!")
                
        self._event_pairs_by_type[event.event_type].append(event)
        self._event_pairs_by_id[event.id_].append(event)
        self._sequential_events.append(event)
        return event.id_

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Store event end data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        for payload in event.payload:
            if payload == EventPayload.FUNCTION_OUTPUT:
                if "Error:" in event.payload[payload]:
                    continue
                print (event.payload[payload])

        self._event_pairs_by_type[event.event_type].append(event)
        self._event_pairs_by_id[event.id_].append(event)
        self._sequential_events.append(event)
        self._trace_map = defaultdict(list)

    def get_events(self, event_type: Optional[CBEventType] = None) -> List[CBEvent]:
        """Get all events for a specific event type."""
        if event_type is not None:
            return self._event_pairs_by_type[event_type]

        return self._sequential_events

    def _get_event_pairs(self, events: List[CBEvent]) -> List[List[CBEvent]]:
        """Helper function to pair events according to their ID."""
        event_pairs: Dict[str, List[CBEvent]] = defaultdict(list)
        for event in events:
            event_pairs[event.id_].append(event)

        return sorted(
            event_pairs.values(),
            key=lambda x: datetime.strptime(x[0].time, TIMESTAMP_FORMAT),
        )

    def _get_time_stats_from_event_pairs(
        self, event_pairs: List[List[CBEvent]]
    ) -> EventStats:
        """Calculate time-based stats for a set of event pairs."""
        total_secs = 0.0
        for event_pair in event_pairs:
            start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
            end_time = datetime.strptime(event_pair[-1].time, TIMESTAMP_FORMAT)
            total_secs += (end_time - start_time).total_seconds()

        return EventStats(
            total_secs=total_secs,
            average_secs=total_secs / len(event_pairs),
            total_count=len(event_pairs),
        )

    def get_event_pairs(
        self, event_type: Optional[CBEventType] = None
    ) -> List[List[CBEvent]]:
        """Pair events by ID, either all events or a specific type."""
        if event_type is not None:
            return self._get_event_pairs(self._event_pairs_by_type[event_type])

        return self._get_event_pairs(self._sequential_events)

    def get_llm_inputs_outputs(self) -> List[List[CBEvent]]:
        """Get the exact LLM inputs and outputs."""
        return self._get_event_pairs(self._event_pairs_by_type[CBEventType.LLM])

    def get_event_time_info(
        self, event_type: Optional[CBEventType] = None
    ) -> EventStats:
        event_pairs = self.get_event_pairs(event_type)
        return self._get_time_stats_from_event_pairs(event_pairs)

    def flush_event_logs(self) -> None:
        """Clear all events from memory."""
        self._event_pairs_by_type = defaultdict(list)
        self._event_pairs_by_id = defaultdict(list)
        self._sequential_events = []

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Launch a trace."""
        self._trace_map = defaultdict(list)
        self._cur_trace_id = trace_id

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Shutdown the current trace."""
        self._trace_map = trace_map or defaultdict(list)
        if self.print_trace_on_end:
            self.print_trace_map()

    def _print_trace_map(self, cur_event_id: str, level: int = 0) -> None:
        """Recursively print trace map to terminal for debugging."""
        event_pair = self._event_pairs_by_id[cur_event_id]
        if event_pair:
            time_stats = self._get_time_stats_from_event_pairs([event_pair])
            indent = " " * level * 2
            self._print(
                f"{indent}|_{event_pair[0].event_type} -> {time_stats.total_secs} seconds",
            )

        child_event_ids = self._trace_map[cur_event_id]
        for child_event_id in child_event_ids:
            self._print_trace_map(child_event_id, level=level + 1)

    def print_trace_map(self) -> None:
        """Print simple trace map to terminal for debugging of the most recent trace."""
        self._print("*" * 10)
        self._print(f"Trace: {self._cur_trace_id}")
        self._print_trace_map(BASE_TRACE_EVENT, level=1)
        self._print("*" * 10)
    
    def _print(self, str) -> None:
        if self.logger:
            self.logger.debug(str)
        else:
            # This branch is to preserve existing behavior.
            print(str, flush=True)

    @property
    def event_pairs_by_type(self) -> Dict[CBEventType, List[CBEvent]]:
        return self._event_pairs_by_type

    @property
    def events_pairs_by_id(self) -> Dict[str, List[CBEvent]]:
        return self._event_pairs_by_id

    @property
    def sequential_events(self) -> List[CBEvent]:
        return self._sequential_events


class GithubSearchTool(BaseToolSpec):
    """
    Uses gitub to search relevant items.
    """
    spec_functions = [
        "find_trending_github_repos",
        "find_relevant_repos",
    ]


    def generate_topics_from_query(self, query: str):
         # Transform embedding to search a DB storing github repo-descriptions.
        class Response(BaseModel):
            output: List[str]
        client = instructor.from_openai(OG_OpenAI())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=Response,
            messages=[
                {"role": "system", "content": """
                    You are an expert at understanding queries of a software engineer and related it with possible github repositories.
                    Your rules is to understand the query and transform into possible github repository topics.
                    Follow these guidelines:
                    1) Do not create length topics more than 5 words.
                    2) Always use the key term in all topics. For example if query is about LLM always use LLM in each topic.
                    3) Do not use any irrelevant terms from the query to make a topic and all topics should be similar in context but
                    different enough to help find all relevant github repos related to the query.
                    4) Do not split key terms to make a new topic. For example Do not do 'LLM finetuning' -> ['LLM', 'finetuning']  
                    5) Make as less topics as possible because using these topics to search Github is expensive and rate limited.

                    Example:
                    query: "relevant github repositories for LLM finetuning."
                    Correct Answer:
                        topics: "['LLM finetuning']"
                    Wrong Answer:
                        topics: "['LLM finetuning', 'LLM', 'finetuning']"

                    """},
                {"role": "user", "content": f"Generate 4 possible topics for query: {query}."},
            ]
        )
        return response


    def check_if_repo_relevant(self, query: str, github_item: GithubRepoItem):
        class Response(BaseModel):
            is_relevant: bool
            reason: str
        
        client = instructor.from_openai(OG_OpenAI())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=Response,
            messages=[
                {"role": "system", "content": """
                    You are helping a software engineer in figuring out if trending repository on github is relevant to a query they are interested in.
                    They have given you a query and a github repo information. Follow these guidelines.
                    1) Understand the query and try to break down into multiple possible topics.
                    2) For each topic, think carefully if the topic is about a particular language or about partcular technique.
                    3) Use all information about github repo to answer Yes or No if the repo can be relevant.
                    4) If you are not totally sure, prefer saying NO.
                 
                    Give reasoning behind your reason of saying YES or NO.

                    Example:
                    query: "LLM"
                    github_repo:
                        description: Data framework for LLM applications
                    
                    Answer:
                        is_relevant: True
                        reason: the repo helps build integrating data into LLM to build applications.
                    """},
                {"role": "user", "content": f"""
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

                 """},
            ]
        )
        return response


    def find_trending_github_repos(self, query : Optional[str]=None) -> AgentResponse:
        """Returns a list storing trending GithubRepoItem containing information about the repos.

        Args:
            query: If provided, then filters the trending repos which have description most similar
                to the query, otherwise returns all the trending repos currently.
        """
        items = GithubDataLoader().get_trending_repos(since="daily", top_k=50)
        if query is not None:
            class RelevantRepos(BaseModel):
                repo: GithubRepoItem
                relevance_reason: str
            
            relevant_items = []
            for item in items:
                response = self.check_if_repo_relevant(query, item)
                if response.is_relevant:
                    relevant_items.append(
                        RelevantRepos(repo=item,relevance_reason=response.reason))
            return AgentResponse(
                message="List of all trending github repos related to query with reason for relevance.",
                content=relevant_items
            )

        return AgentResponse(
            message="All trending github repos currently.",
            content=items
        )

    def find_relevant_repos(self, query: str) -> AgentResponse:
        """Given a query, finds all relevant topics which can be searched on github and returns relevant github repos."""
       
        topics = self.generate_topics_from_query(query)
        
        items = GithubDataLoader().search_repos_by_topic(topics=topics, min_stars=0, top_k=10)
        return AgentResponse(
            message=f"""
            Relevant repos for {query} based on the following topics generated:
            {topics}
            """,
            content=items
        )


class InternalDatabaseSearch(BaseToolSpec):
    """
    Uses a maintained vector db to search for relevant items given query.
    """
    spec_functions = [
        "retrieve_relevant_repos",
        "add_github_repos"
    ]

    def __init__(self, metadata_db_config: DBConfig, code_db_config: DBConfig, github_access_token: str):
        self.client = instructor.from_openai(OG_OpenAI())
        
        self.metadata_db = VectorDB(metadata_db_config)
        self.code_db = VectorDB(code_db_config)
        self.github_access_token = github_access_token

    
    def retrieve_relevant_repos(self, query: str, top_k:int=5) -> AgentResponse:
        """Retrieve top k relevant code repos given the query."""
        # Transform embedding to search a DB storing github repo-descriptions.
        class Questions(BaseModel):
            questions: List[str]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=Questions,
            messages=[
                {"role": "system", "content": """
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
                 """},
                {"role": "user", "content": f"Generate 4 descriptions for query: {query}"},
            ]
        )
        questions = response.questions
        response = self.metadata_db.search(
            query_texts=questions,
            top_k=top_k
        )
        if not len(response['metadatas']):
            return AgentResponse(
                message="""
                No relevant code repos found for query: {query} based on retrieval from following generated descriptions:

                {questions}
                """,
                content=None
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
        for i in range(len(response['metadatas'])):
            print(response['metadatas'][i])
            relevant_items.append(
                RelevantRepoItem(summary=response['documents'][i][0],
                    repo_url=response['metadatas'][i][0]['repo_url'],
                    desc=response['metadatas'][i][0]['desc'],
                    lang=response['metadatas'][i][0]['lang'],
                    name=response['metadatas'][i][0]['name'],
                    owner=response['metadatas'][i][0]['owner'],
                    stars=response['metadatas'][i][0]['stars'],
                    forks=response['metadatas'][i][0]['forks']))

        return AgentResponse(
            message="""
            Found code repos for query: {query} based on following generated descriptions:

            {questions}
            """,
            content=relevant_items
        )
        

    def generate_repo_summary(self, prompt: str) -> str:
        class SummaryResponse(BaseModel):
            github_url: str
            summary: str
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            response_model=SummaryResponse,
            messages=[
                {"role": "system", "content": """
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
                 """},
                {"role": "user", "content": f"""{prompt}"""},
            ]
        )
        return response.summary

    def add_github_repos(self, github_urls: List[str]) -> AgentResponse:
        """Updates the internal database if any github url in the provided list is not present."""
        added_items = []
        present_items = []

        for github_url in github_urls:        

            metadata_nodes = self.metadata_db.filter({"repo_url": github_url})
            code_nodes = self.code_db.filter({"repo_url": github_url})

            if len(metadata_nodes['metadatas']) and len(code_nodes['metadatas']):
                del code_nodes
                for i in range(len(metadata_nodes['metadatas'])):
                    present_items.append(metadata_nodes['metadatas'][i])
                    present_items[-1].update({'summary': metadata_nodes['documents'][i]})
            else:
                match = re.search(r"github\.com/([^/]+/[^/]+)", github_url)
                if not match:
                    continue
                github_item = GithubDataLoader().search_repos_by_name([match.group(1)])[0]
                nodes, code_documents = chunk_repo(github_item, self.github_access_token)
                tool_map_dict = CodeHierarchyNodeParser.get_code_hierarchy_from_nodes(
                    nodes, max_depth=0
                )[0]
                num_tries = 3
                max_depth_to_consider = 3
                summary = None
                while num_tries:
                    tool_map = tool_map_to_markdown(tool_map_dict, 0, max_depth=max_depth_to_consider)
                    
                    summary_prompt = f"""
                    
                    Repository Details:

                    URL: {github_item.repo_url}
                    Descripption: {github_item.desc}
                    Language: {github_item.lang}
                    Name: {github_item.name}
                    Owner: {github_item.owner}
                    Stars: {github_item.stars}
                    Forks: {github_item.forks}

                    Here is how the repository structure looks like with folder, module names present etc.

                    {tool_map}
                    """

                    try:
                        summary = self.generate_repo_summary(summary_prompt)
                        break
                    except Exception as e:
                        max_depth_to_consider -= 1
                    num_tries -= 1
                assert summary is not None
                self.metadata_db.add(documents=[summary], metadatas=[github_item.dict()], ids=[github_url])
                documents = [doc.text for doc in code_documents]
                doc_metadata = [doc.dict() for doc in code_documents]
                for m in doc_metadata:
                    m.pop('text', None)
                    cur_keys = list(m.keys())
                    for k in cur_keys:
                        if m[k] is None:
                            m.pop(k)
                        elif isinstance(m[k], dict):
                            m[k] = json.dumps(m[k])
                        elif isinstance(m[k], list):
                            m[k] = json.dumps(m[k])
    
                self.code_db.add(documents=documents,ids=list(map(lambda x : f"{github_url}_{x}", range(len(code_documents)))),
                                  metadatas=list(map(lambda x : {**github_item.dict(), "node_id": x, **doc_metadata[x]}, range(len(code_documents)))))
                added_items.append(github_item.dict())
                added_items[-1].update({'summary': summary})
        if added_items:
            return AgentResponse(message=f"New github URLs are now added to the internal database. Please try using them again", content=added_items)
        return AgentResponse(message=f"All github URLs are already present in the database. They are ready to use", content=present_items)
        

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
        self.llm = OpenAI(temperature=0, model_name=GPT_MODEL_NAME, api_key=OPENAI_API_KEY)
        self.code_db = VectorDB(db_config)
    
    @classmethod
    def get_repo_desc(cls, repo_url):
        github_url_pattern = r'https://github\.com/([^/]+/[^/]+)'
        import re
        match = re.search(github_url_pattern, repo_url)
        name = match.group(1)
        name = name.replace('.git', '')
        url = f"https://api.github.com/search/repositories?q={name}"
        import requests
        func = getattr(requests, "get")
        response = func(url, headers={}, timeout=50)
        if response.status_code != 200:
            assert f"Unable to retrieve {url}"
        response = response.json()
        desc = None
        for item in response["items"]:
            if item.get("stargazers_count", 0) < 100 or not item.get("full_name", "") or not item.get("description", "") or not item.get("language", ""):
                continue
            desc = "".join(item["description"]).strip()
            break
        return desc if desc is not None else ""
    

    def make_nodes_from_documents(self, chroma_nodes):
        documents = []
        for doc,metadata in zip(chroma_nodes['documents'], chroma_nodes['metadatas']):
            documents.append(
                Document(
                    text=doc,
                    id_=metadata['id_'],
                    relationships=json.loads(metadata['relationships']),
                    class_name=metadata['class_name'],
                    excluded_embed_metadata_keys=json.loads(metadata['excluded_embed_metadata_keys']),
                    excluded_llm_metadata_keys=json.loads(metadata['excluded_llm_metadata_keys']),
                    metadata=json.loads(metadata['metadata']),
                    metadata_seperator=metadata['metadata_seperator'],
                    node_id=metadata['node_id'],
                    text_template=metadata['text_template'],
                    metadata_template=metadata['metadata_template'],
                )
            )
        return CodeHierarchyNodeParser(
            language="python",
            # You can further parameterize the CodeSplitter to split the code
            # into "chunks" that match your context window size using
            # chunck_lines and max_chars parameters, here we just use the defaults
            # code_splitter=CodeSplitter(language="python"),
        ).get_nodes_from_documents(documents)
    
    def make_agent(self, github_repo_urls: List[str]) -> AgentResponse:
        """
        Given a list of github repo urls, create a list of openAI agents to chat with.
        """

        def _make_agent(repo_url, nodes):
            query_engine = CodeHierarchyKeywordQueryEngine(
                nodes=nodes,
            )
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name="code_lookup",
                description=f"Useful for looking up information about the {repo_url} codebase.",
            )
            tool_instructions = query_engine.get_tool_instructions()
            system_prompt = f"""

            You are principal software engineer with deep understanding of the following gitub
            repo:
            repo_url: {repo_url}
            repo_description: {self.get_repo_desc(repo_url)}
            
            You are onboarding an early AI engineer to understand the the repo's codebase. 
            They will give you the query for a repo. You have to access to a code lookup tool 
            for the github repo and here are the instructions to use it:

            {tool_instructions}

            Always mention the most relevant pieces of code in your response.

            """
            return OpenAIAgent.from_tools(
                [tool], llm=self.llm, system_prompt=system_prompt, verbose=False
            )

        for repo_url in set(github_repo_urls):
            if repo_url in self._agents:
                continue
            nodes = self.code_db.filter({"repo_url": repo_url})
            if not len(nodes['metadatas']):
                return AgentResponse(message=f"""
                Following github URLs are not present in the database. Please index them into the internal database first using 
                add_github_repos tool from InternalDatabaseSearch Agent.
                """, content=repo_url)

            nodes = self.make_nodes_from_documents(nodes)

            if not len(nodes):
                return AgentResponse(message=f"""
                Not able to process data for {repo_url} from internal database. Gently respond the user that we are not able
                to process this repo url to provide any information related to it.
                """, content=repo_url)
            self._agents[repo_url] = _make_agent(repo_url, nodes)
        
        return AgentResponse(message=f"Created agents for all github urls. You can now ask any queries related to them.", content=None)
    
    @traceable
    def chat(self, repo_url: str, query: str) -> AgentResponse:
        """
        Given a repo_url answer the query by chatting with an agent for that repo.
        """
        agent_response = self.make_agent([repo_url])
        if agent_response.content is not None:
            return agent_response
        
        response = self._agents[repo_url].chat(query)

        code_chunks = ""

        for source in response.sources:
            if isinstance(source.raw_output, Response) and source.tool_name == "code_lookup" and source.raw_output.response:
                code_chunks += source.raw_output.response
                code_chunks += "\n\n"
        response_str = str(response)
        
        repo_desc = self.get_repo_desc(repo_url)
        messages = []
        messages.append(ChatMessage(role="assistant",content= f"""
        You are principal software engineer with deep understanding of the
        provided github repo. you are onboarding an early AI engineer to understand the
        complexities that repo's codebase. They will give
        you the URL, description and some relevant code from that repo. Think step by step
        using the following guidelines and give a detailed analysis to help onboard. Mention
        relevant code while answering if needed to explain better but make sure to not fill
        the response only with code. try to give explainations. 
        
        If you see 'Code replaced for brevity' then a uuid, in the code-chunks, that means
        the code is cut-out there, so make sure to not mention the UUID in the response.
        """))
        messages.append(ChatMessage(role="user",content=f"""
        Hi, Can you help understand code related to my query from a gitub repo.
        I am listing the URL, github repo description, output from a code lookup tool: 
                                    
                                    
        Query: {query}
        
        URL: {repo_url}
        Description: {repo_desc}
        Code Lookup Tool Output: {response_str}
        """))
        return AgentResponse(message=response.sources, 
                             content=str(response))


    def compare(self, repo_a_url: str, repo_b_url: str, query: str) -> AgentResponse:
        """
        Given a query compares the code in two repos and yield a response
        """
        if repo_a_url == repo_b_url:
            return AgentResponse(
                message="Please ask user to pass valid inputs for comparison. Both github repo URLs are identical and can't do comparison on identical things.",
                content=None
            )
        agent_response = self.make_agent([repo_a_url, repo_b_url])
        if agent_response.content is not None:
            return agent_response
        
        
        response_a = str(self._agents[repo_a_url].chat(query))
        response_b = str(self._agents[repo_b_url].chat(query))
        # compare the response from both repos and construct the output.
        repo_desc_a = self.get_repo_desc(repo_a_url)
        repo_desc_b = self.get_repo_desc(repo_b_url)

        messages = []
        messages.append(ChatMessage(role="assistant",content= f"""
        You are principal software engineer with deep understanding of python and
        machine learning. you are onboarding an early AI engineer to understand the
        complexities and difference between two github codebases. They will give
        you the URL, description and some code from each repo. Think step by step
        using the following guidelines and give a detailed analysis to help onboard:
        1) Explain each repo's provided code and what is required to use it.
        2) Are both code-chunks doing the same thing? 
        3) How is one repo's code better than other?
        """))
        messages.append(ChatMessage(role="user",content=f"""
        Here are the details of two repos, help me compare them and analyze both
        based on my query. I am listing the URL, description and output from a
        code lookup tool related to my query: {query}
        
        Repo 1:
                                    
        URL: {repo_a_url}
        Description: {repo_desc_a}
        Code Lookup Tool Output: {response_a}
        

        Repo 2:
                                    
        URL: {repo_b_url}
        Description: {repo_desc_b}
        Code Lookup Tool Output: {response_b}
        """))
        return AgentResponse(message=f"""
            Chat response for query: {query} based on relevant code chunks found from both code repos:
            {response_a}

            {response_b}
            """, content=self.llm.chat(messages).message.content)



def create_agent():
    llm = OpenAI(temperature=0, model=GPT_MODEL_NAME)
    llama_debug = CustomCallbackHandler(print_trace_on_end=False)
    callback_manager = CallbackManager([llama_debug])
    github_tool_spec = GithubSearchTool()
    code_analyzer_tool_spec = CodeAnalyzerTool()
    load_and_search_tool_spec = InternalDatabaseSearch(metadata_db_path="./metadata_db", 
                                                       codes_db_path="./code_db",
                                                       github_access_token=GITHUB_TOKEN)
    code_analyzer_agent = OpenAIAgent.from_tools(code_analyzer_tool_spec.to_tool_list(), 
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
        """, llm=llm, verbose=True, callback_manager=None,
        max_function_calls=100)
    github_search_agent = OpenAIAgent.from_tools(github_tool_spec.to_tool_list(), 
        system_prompt="""
        Github is world's largest platform for storing, tracking and collaborating
        on various types of software projects. Understand the context, to figure
        out which tool can answer the query best.
        Note: Searching topics on github works well if the topics are concise and
        are at max 2 words. Depending on the context, break into multiple small topics. 
        """, llm=llm, verbose=True,
        max_function_calls=100)
    load_and_search_agent = OpenAIAgent.from_tools(load_and_search_tool_spec.to_tool_list(),
        system_prompt="""
        Depending on the query, find any relevant github repos inside the database or ingest
        the repos for further use. Searching external is costly, and we want to use this
        internal database as much as possible to be more efficient.
        """, llm=llm, verbose=True,
        max_function_calls=100)
    code_analyzer_query_agent = QueryEngineTool.from_defaults(
        code_analyzer_agent,
        name='code_analyzer_query_agent',
        description="""
        Given a  1) question/query  2) github repo URLs, 
        Answers or compares using relevant context from gitub repository codebase
        of provide github URLs for given a question/query.
        """)
    github_search_query_agent = QueryEngineTool.from_defaults(
        github_search_agent,
        name="github_search_query_agent",
        description="""
        Use this to search github for relevant or trending repos.
        """   
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
        """
    )

    return OpenAIAgent.from_tools(
        [code_analyzer_query_agent, github_search_query_agent, load_and_search_query_agent],
        llm=llm,verbose=True,
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
        callback_manager=callback_manager,
        max_function_calls=100
    )


if __name__ == "__main__":
    query = sys.argv[1]
    agent = create_agent()
    print(agent.chat(query))
