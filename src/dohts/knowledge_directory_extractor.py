from typing import List
from langchain_community.document_loaders import DirectoryLoader


class KnowledgeDirectoryExtractor:

    @staticmethod
    def directory_loader(knowledge_dir: str) -> List:
        """
            Loads all the docs that are in a directory
        """
        print(knowledge_dir)
        loader = DirectoryLoader(knowledge_dir, glob="**/*.pdf")
        docs: List = loader.load()
        return docs
