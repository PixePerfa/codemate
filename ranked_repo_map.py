import colorsys
import os
import random
import sys
import shutil
import warnings
import networkx as nx
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm

from tree_sitter_languages import get_language, get_parser  # noqa: E402


from llama_index.core.readers.file.base import SimpleDirectoryReader
from github import Github
from git import Repo
from urllib.parse import urlparse
from rich import print

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)


Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


QUERY_PATH = "./queries"
GITHUB_TOKEN = "xxxxx"

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


def clone_repo(github_url, github_access_token):
    """
    Clone the GitHub repository to a local directory
    """
    g = Github(github_access_token) if github_access_token else Github()

    def extract_repo_name(github_url):
        path = urlparse(github_url).path
        repo_name = path.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name

    def extract_repo_details(url):
        path = urlparse(url).path.lstrip("/")
        repo_owner, repo_name = path.split("/")
        repo_name = repo_name.rstrip(".git")
        return repo_owner, repo_name

    repo_owner, repo_name = extract_repo_details(github_url)
    local_dir = f"/tmp/{extract_repo_name(github_url)}"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
        print(f"Directory '{local_dir}' deleted.")

    try:
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        default_branch = repo.default_branch
    except Exception:
        default_branch = "main"

    try:
        Repo.clone_from(github_url, local_dir)
    except Exception:
        print(f"Directory {local_dir} is not empty.")

    return local_dir, default_branch, None


def get_documents(github_url, lang, return_files=False):
    repo_url = github_url
    repo_lang = lang
    assert repo_lang
    repo_lang = repo_lang.lower()
    repo_lang = repo_lang if repo_lang != "c++" else "cpp"
    lang_extensions = LANG_EXTENSIONS_DICT.get(repo_lang, [])
    if not lang_extensions:
        return None, None
    try:
        local_dir, _, _ = clone_repo(repo_url, GITHUB_TOKEN)
    except Exception:
        return []
    valid_files = []
    for dirpath, dirnames, filenames in os.walk(local_dir):
        # ignore dotfolders
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for file in filenames:
            _, file_extension = os.path.splitext(file)
            if file_extension.lower() not in lang_extensions:
                continue
            full_path = os.path.join(dirpath, file)
            # ignore folders belonging to tests, legacy
            if (
                "test" in full_path
                or "legacy" in full_path
                or ".github" in full_path
                or "mock" in full_path
            ):
                continue
            valid_files.append(Path(full_path))

    if return_files:
        return valid_files
    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_metadata=lambda x: {"filepath": x},
    ).load_data()

    shutil.rmtree(local_dir)
    return documents


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    cache_missing = False

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()

        self.max_map_tokens = map_tokens

        self.repo_content_prefix = repo_content_prefix

    def token_count(self, text):
        return 0
        # if type(text) is str:
        #     msgs = text
        # else:
        #     msgs = json.dumps(text)

        # return len(litellm.encode(model="gpt-3.5-turbo", text=msgs))

    def get_repo_map(self, chat_files, other_files):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return

        try:
            files_listing = self.get_ranked_tags_map(chat_files, other_files)
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        num_tokens = self.token_count(files_listing)
        if self.verbose:
            self.io.tool_output(f"Repo-map: {num_tokens/1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        if not path.exists():
            self.cache_missing = True
        self.TAGS_CACHE = Cache(path)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        if (
            cache_key in self.TAGS_CACHE
            and self.TAGS_CACHE[cache_key]["mtime"] == file_mtime
        ):
            return self.TAGS_CACHE[cache_key]["data"]

        # miss!

        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache
        self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
        self.save_tags_cache()
        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)

        # Load the tags queries
        try:
            scm_fname = os.path.join(QUERY_PATH, f"tree-sitter-{lang}-tags.scm")
        except KeyError:
            return

        fp = open(scm_fname, "r")
        query_scm = fp.read()

        code = open(fname, "r").read()
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(self, chat_fnames, other_fnames):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        if self.cache_missing:
            fnames = tqdm(fnames)
        self.cache_missing = False

        for fname in fnames:
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it no longer exists"
                        )

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = 1.0
                chat_rel_fnames.add(rel_fname)

            tags = list(self.get_tags(fname, rel_fname))

            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                if tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))
        # import pdb
        # pdb.set_trace()

        G = nx.MultiDiGraph()

        for ident in idents:
            definers = defines[ident]
            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # if referencer == definer:
                    #    continue
                    G.add_edge(referencer, definer, weight=num_refs, ident=ident)

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked[src]
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: x[1]
        )

        # import pdb
        # pdb.set_trace()

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            # print(f"{rank:.03f} {fname} {ident}")
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
        # import pdb
        # pdb.set_trace()
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        # import pdb
        # pdb.set_trace()

        return ranked_tags

    def get_ranked_tags_map(self, chat_fnames, other_fnames=None):
        if not other_fnames:
            other_fnames = list()

        ranked_tags = self.get_ranked_tags(chat_fnames, other_fnames)
        num_tags = len(ranked_tags)
        # print(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None

        chat_rel_fnames = [self.get_rel_fname(fname) for fname in chat_fnames]

        while lower_bound <= upper_bound:
            middle = (lower_bound + upper_bound) // 2
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            if num_tokens < self.max_map_tokens:
                best_tree = tree
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1
        # import pdb
        # pdb.set_trace()
        return best_tree

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        context = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if context:
                    context.add_context()
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += context.format()
                    context = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"

                if type(tag) is Tag:
                    code = open(tag.fname, "r").read() or ""

                    context = TreeContext(
                        tag.rel_fname,
                        code,
                        color=False,
                        line_number=False,
                        child_context=False,
                        last_line=False,
                        margin=0,
                        mark_lois=False,
                        loi_pad=0,
                        # header_max=30,
                        show_top_of_file_parent_scope=False,
                    )
                cur_fname = this_rel_fname

            if context:
                context.add_lines_of_interest([tag.line])

        return output


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


if __name__ == "__main__":
    lang = sys.argv[1]
    if lang == "python":
        repoItem = {
            "github_url": "https://github.com/OpenDevin/OpenDevin",
            "lang": "python",
            "return_files": True,
        }
    elif lang == "rust":
        repoItem = {
            "github_url": "https://github.com/rust-lang/rustlings",
            "lang": "rust",
            "return_files": True,
        }
    docs = get_documents(**repoItem)
    if lang == "python":
        chat_files = [doc for doc in docs if "opendevin/memory/memory.py" in str(doc)]
        other_files = [
            doc for doc in docs if "opendevin/memory/memory.py" not in str(doc)
        ]
    elif lang == "rust":
        chat_files = [
            doc for doc in docs if "exercises/02_functions/functions2.rs" in str(doc)
        ]
        other_files = [
            doc
            for doc in docs
            if "exercises/02_functions/functions2.rs" not in str(doc)
        ]

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_files, other_files)

    print(repo_map)
