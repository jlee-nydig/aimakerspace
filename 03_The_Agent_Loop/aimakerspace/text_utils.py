import os
from typing import List, Dict, Optional, Tuple

import pymupdf


class PDFFileLoader:
    def __init__(self, path: str):
        self.documents = []
        self.metadata = []  # Track metadata for each document
        self.path = path

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        doc = pymupdf.open(self.path)
        text = ""
        for page in doc:
            text += page.get_text()
        self.documents.append(text)

        # Track metadata
        filename = os.path.basename(self.path)
        self.metadata.append({
            "source": filename,
            "path": self.path,
            "num_pages": len(doc)
        })

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in sorted(files):  # Sort for consistent ordering
                if file.endswith(".pdf"):
                    filepath = os.path.join(root, file)
                    doc = pymupdf.open(filepath)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    self.documents.append(text)

                    # Track metadata
                    self.metadata.append({
                        "source": file,
                        "path": filepath,
                        "num_pages": len(doc)
                    })

    def load_documents(self):
        self.load()
        return self.documents

    def load_documents_with_metadata(self) -> List[Tuple[str, Dict]]:
        """Load documents and return them with their metadata."""
        self.load()
        return list(zip(self.documents, self.metadata))


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

    def split_texts_with_metadata(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Split texts while preserving metadata for each chunk.

        Args:
            texts: List of text strings to split
            metadatas: Optional list of metadata dicts (one per text)

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]

        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")

        chunks_with_metadata = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.split(text)
            for chunk_idx, chunk in enumerate(chunks):
                # Create metadata for this chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                chunk_metadata["total_chunks"] = len(chunks)

                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })

        return chunks_with_metadata


class RecursiveTextSplitter:
    """
    More sophisticated text splitter that respects natural language boundaries.

    Tries to split on separators in order of preference:
    1. Double newlines (paragraphs)
    2. Single newlines (lines)
    3. Spaces (words)
    4. Characters (as last resort)

    This preserves semantic meaning better than pure character splitting.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators in order of preference
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters (fallback)
        ]

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a specific separator."""
        if separator:
            return text.split(separator)
        # If separator is empty string, split into characters
        return list(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size."""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If adding this split would exceed chunk_size
            if current_length + split_len + len(separator) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    # Keep removing items from the start until we're under overlap size
                    overlap_length = 0
                    overlap_chunks = []
                    for item in reversed(current_chunk):
                        overlap_length += len(item) + len(separator)
                        if overlap_length > self.chunk_overlap:
                            break
                        overlap_chunks.insert(0, item)

                    current_chunk = overlap_chunks
                    current_length = sum(len(c) for c in current_chunk) + \
                                   len(separator) * max(0, len(current_chunk) - 1)

            # Add current split
            current_chunk.append(split)
            current_length += split_len + (len(separator) if current_chunk else 0)

        # Add remaining chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def split(self, text: str) -> List[str]:
        """
        Recursively split text using different separators.

        Tries each separator in order, using the first one that produces
        reasonable chunks.
        """
        # Try each separator
        for i, separator in enumerate(self.separators):
            # Split by current separator
            splits = self._split_text_with_separator(text, separator)

            # Check if any split is too large
            max_split_len = max(len(s) for s in splits) if splits else 0

            # If all splits are small enough, merge them
            if max_split_len <= self.chunk_size:
                return self._merge_splits(splits, separator)

            # If we've tried all separators except the last (character-level)
            if i == len(self.separators) - 1:
                # Force character-level splitting
                return self._merge_splits(splits, separator)

            # Otherwise, try recursively splitting the large chunks
            good_splits = []
            for split in splits:
                if len(split) <= self.chunk_size:
                    good_splits.append(split)
                else:
                    # Recursively split this chunk with remaining separators
                    subsplitter = RecursiveTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=self.separators[i+1:]
                    )
                    good_splits.extend(subsplitter.split(split))

            return self._merge_splits(good_splits, separator)

        return [text]

    def split_texts(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks."""
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

    def split_texts_with_metadata(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Split texts while preserving metadata for each chunk.

        Args:
            texts: List of text strings to split
            metadatas: Optional list of metadata dicts (one per text)

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]

        if len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadatas must match")

        chunks_with_metadata = []

        for text, metadata in zip(texts, metadatas):
            chunks = self.split(text)
            for chunk_idx, chunk in enumerate(chunks):
                # Create metadata for this chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_idx
                chunk_metadata["total_chunks"] = len(chunks)

                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })

        return chunks_with_metadata


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
