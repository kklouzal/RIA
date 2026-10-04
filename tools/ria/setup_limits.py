"""Shared setup admission ceilings, separate from measured DSER/runtime limits."""

# Compact packages permit4096 one-GiB shards plus their indices/pages/metadata.
# Per-job settings must set a smaller appropriate byte quota and wall budget.
MAX_TREE = 5 << 40
MAX_FILES = 16384
MAX_MESSAGE = 8 << 20
MAX_NODES = 131072
CHUNK = 4 << 20
MAX_TRANSFER_SECONDS = 86400
MAX_SETUP_SECONDS = 604800
MAX_RECEIPT = 1 << 20
MAX_CHUNK_METADATA = 64 << 10
