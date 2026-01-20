"""
Compression for large task data.

Implements adv-perf-006: Compression for large task data

Features:
- Automatic compression based on data size
- Multiple compression algorithms (gzip, zlib, lz4)
- Transparent compression/decompression
- Memory-efficient streaming compression
"""

import gzip
import zlib
import json
import base64
from typing import Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import io

# Try to import lz4 for faster compression
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class CompressionConfig:
    """Configuration for compression behavior."""
    # Minimum size in bytes before compressing
    min_size_threshold: int = 1024  # 1KB

    # Default algorithm
    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP

    # Compression level (1-9 for gzip/zlib, ignored for lz4)
    level: int = 6

    # Whether to encode as base64 for JSON storage
    base64_encode: bool = True


class CompressedData:
    """
    Wrapper for compressed data that handles serialization.

    Can be serialized to JSON and maintains metadata about
    the compression used.
    """

    def __init__(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm,
        original_size: int,
        compressed_size: int,
        base64_encoded: bool = False
    ):
        self.data = data
        self.algorithm = algorithm
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.base64_encoded = base64_encoded

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.compressed_size / self.original_size if self.original_size > 0 else 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = self.data
        if not self.base64_encoded:
            data = base64.b64encode(data).decode("ascii")

        return {
            "_compressed": True,
            "algorithm": self.algorithm.value,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "data": data if isinstance(data, str) else data.decode("ascii"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompressedData":
        """Create from dictionary."""
        data = base64.b64decode(d["data"])
        return cls(
            data=data,
            algorithm=CompressionAlgorithm(d["algorithm"]),
            original_size=d["original_size"],
            compressed_size=d["compressed_size"],
            base64_encoded=False,
        )


def compress_data(
    data: Union[str, bytes, dict, list],
    config: Optional[CompressionConfig] = None
) -> Union[CompressedData, Any]:
    """
    Compress data if it exceeds the threshold.

    Args:
        data: Data to compress (string, bytes, or JSON-serializable object)
        config: Compression configuration

    Returns:
        CompressedData if compressed, original data otherwise
    """
    if config is None:
        config = CompressionConfig()

    # Convert to bytes if needed
    if isinstance(data, (dict, list)):
        data_bytes = json.dumps(data).encode("utf-8")
    elif isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data

    original_size = len(data_bytes)

    # Skip compression for small data
    if original_size < config.min_size_threshold:
        return data

    # Compress based on algorithm
    if config.algorithm == CompressionAlgorithm.GZIP:
        compressed = gzip.compress(data_bytes, compresslevel=config.level)
    elif config.algorithm == CompressionAlgorithm.ZLIB:
        compressed = zlib.compress(data_bytes, level=config.level)
    elif config.algorithm == CompressionAlgorithm.LZ4:
        if not LZ4_AVAILABLE:
            # Fall back to gzip
            compressed = gzip.compress(data_bytes, compresslevel=config.level)
            config.algorithm = CompressionAlgorithm.GZIP
        else:
            compressed = lz4.frame.compress(data_bytes)
    else:
        return data

    compressed_size = len(compressed)

    # Only use compression if it actually reduces size
    if compressed_size >= original_size:
        return data

    if config.base64_encode:
        compressed = base64.b64encode(compressed)

    return CompressedData(
        data=compressed,
        algorithm=config.algorithm,
        original_size=original_size,
        compressed_size=compressed_size,
        base64_encoded=config.base64_encode,
    )


def decompress_data(
    data: Union[CompressedData, dict, Any]
) -> Union[str, bytes, dict, list]:
    """
    Decompress data if it's compressed.

    Args:
        data: Compressed data or original data

    Returns:
        Decompressed data
    """
    # Handle dictionary format (from JSON)
    if isinstance(data, dict) and data.get("_compressed"):
        data = CompressedData.from_dict(data)

    # If not compressed, return as-is
    if not isinstance(data, CompressedData):
        return data

    # Decode base64 if needed
    compressed_bytes = data.data
    if data.base64_encoded:
        compressed_bytes = base64.b64decode(compressed_bytes)

    # Decompress based on algorithm
    if data.algorithm == CompressionAlgorithm.GZIP:
        decompressed = gzip.decompress(compressed_bytes)
    elif data.algorithm == CompressionAlgorithm.ZLIB:
        decompressed = zlib.decompress(compressed_bytes)
    elif data.algorithm == CompressionAlgorithm.LZ4:
        if not LZ4_AVAILABLE:
            raise RuntimeError("LZ4 compression not available")
        decompressed = lz4.frame.decompress(compressed_bytes)
    else:
        return compressed_bytes

    # Try to decode as JSON
    try:
        return json.loads(decompressed.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        try:
            return decompressed.decode("utf-8")
        except UnicodeDecodeError:
            return decompressed


class CompressedStorage:
    """
    Storage wrapper that automatically compresses large data.

    Example:
        storage = CompressedStorage(Path("tasks.json.gz"))
        storage.write(large_task_data)
        data = storage.read()
    """

    def __init__(
        self,
        path: Path,
        config: Optional[CompressionConfig] = None,
        auto_compress_files: bool = True
    ):
        self.path = Path(path)
        self.config = config or CompressionConfig()
        self.auto_compress_files = auto_compress_files

    def write(self, data: Any) -> int:
        """
        Write data to file with automatic compression.

        Returns:
            Number of bytes written
        """
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, indent=2, default=str)
            data_bytes = json_str.encode("utf-8")
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data

        if self.auto_compress_files and len(data_bytes) >= self.config.min_size_threshold:
            # Write as gzipped file
            write_path = self.path.with_suffix(self.path.suffix + ".gz")
            with gzip.open(write_path, "wb", compresslevel=self.config.level) as f:
                f.write(data_bytes)
            return len(data_bytes)
        else:
            with open(self.path, "wb") as f:
                f.write(data_bytes)
            return len(data_bytes)

    def read(self) -> Any:
        """Read and decompress data from file."""
        # Check for compressed version first
        gz_path = self.path.with_suffix(self.path.suffix + ".gz")
        if gz_path.exists():
            with gzip.open(gz_path, "rb") as f:
                data_bytes = f.read()
        elif self.path.exists():
            with open(self.path, "rb") as f:
                data_bytes = f.read()
        else:
            raise FileNotFoundError(f"No file found at {self.path} or {gz_path}")

        # Try to parse as JSON
        try:
            return json.loads(data_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return data_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return data_bytes

    def exists(self) -> bool:
        """Check if the file exists (compressed or not)."""
        gz_path = self.path.with_suffix(self.path.suffix + ".gz")
        return self.path.exists() or gz_path.exists()


class StreamingCompressor:
    """
    Memory-efficient streaming compression for very large data.

    Example:
        with StreamingCompressor("output.gz") as compressor:
            for chunk in large_data_source:
                compressor.write(chunk)
    """

    def __init__(
        self,
        path: str,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        level: int = 6
    ):
        self.path = path
        self.algorithm = algorithm
        self.level = level
        self._file = None
        self._compressor = None
        self._bytes_written = 0
        self._bytes_compressed = 0

    def __enter__(self) -> "StreamingCompressor":
        self._file = open(self.path, "wb")

        if self.algorithm == CompressionAlgorithm.GZIP:
            self._compressor = gzip.GzipFile(
                fileobj=self._file,
                mode="wb",
                compresslevel=self.level
            )
        elif self.algorithm == CompressionAlgorithm.ZLIB:
            self._compressor = zlib.compressobj(level=self.level)
        elif self.algorithm == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
            self._compressor = lz4.frame.LZ4FrameCompressor()
            self._file.write(self._compressor.begin())

        return self

    def write(self, data: Union[str, bytes]) -> int:
        """Write data chunk."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        self._bytes_written += len(data)

        if self.algorithm == CompressionAlgorithm.GZIP:
            self._compressor.write(data)
        elif self.algorithm == CompressionAlgorithm.ZLIB:
            compressed = self._compressor.compress(data)
            self._file.write(compressed)
        elif self.algorithm == CompressionAlgorithm.LZ4:
            compressed = self._compressor.compress(data)
            self._file.write(compressed)

        return len(data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._compressor:
            if self.algorithm == CompressionAlgorithm.GZIP:
                self._compressor.close()
            elif self.algorithm == CompressionAlgorithm.ZLIB:
                self._file.write(self._compressor.flush())
            elif self.algorithm == CompressionAlgorithm.LZ4:
                self._file.write(self._compressor.flush())

        if self._file:
            self._bytes_compressed = self._file.tell()
            self._file.close()

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio after closing."""
        if self._bytes_written == 0:
            return 1.0
        return self._bytes_compressed / self._bytes_written
