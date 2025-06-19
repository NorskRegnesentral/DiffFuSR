from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="isp-uv-es/opensr-test",
    repo_type="dataset",
    local_dir="opensr_test_100",
    allow_patterns=["100/*"]
)