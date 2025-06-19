from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="isp-uv-es/opensr-test",
    repo_type="dataset",
    local_dir="load/opensrtest",
    allow_patterns=["100/*"]
)