#!/usr/bin/env python3
"""
批量下载 HuggingFace 上 lerobot 组织的所有数据集到本地 /public 目录。

特性：
  - 自动重试机制，处理速率限制和网络错误
  - 支持指数退避等待（速率限制时）
  - 支持 HuggingFace 认证令牌来提高速率限制
  - 降低并发数和添加请求间隔避免被限流
  - 可选：下载后同步到远端服务器（--remote）

依赖：
  pip install huggingface_hub

使用 HF_TOKEN 的方式：
  1. 通过环境变量：export HF_TOKEN="your_token"
  2. 通过命令行参数：--hf-token "your_token"
  获取令牌：https://huggingface.co/settings/tokens
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, snapshot_download, login


def fetch_lerobot_dataset_ids(api: HfApi) -> List[str]:
    """获取 lerobot 组织下的全部数据集 ID。"""
    ids: List[str] = []
    for ds in api.list_datasets(author="lerobot", full=False):
        ids.append(ds.id)
    return ids


def download_one(dataset_id: str, work_dir: Path) -> Path:
    """将单个数据集下载到本地工作目录，带重试机制。"""
    target_dir = work_dir / dataset_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {dataset_id} -> {target_dir}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            local_path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=2,  # 降低并发数避免被限流
            )
            return Path(local_path)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                # 速率限制，需要等待更长时间
                wait_time = 60 * (2 ** attempt)  # 指数退避：60s, 120s, 240s
                print(f"  触发速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
            elif "snapshot folder" in str(e):
                # 尝试清理后重新下载
                print(f"  快照文件夹错误，清理并重新下载...")
                try:
                    shutil.rmtree(target_dir)
                except:
                    pass
                if attempt < max_retries - 1:
                    continue
            
            if attempt == max_retries - 1:
                raise
            else:
                print(f"  下载失败 (尝试 {attempt + 1}/{max_retries})，等待 30 秒后重试...")
                time.sleep(30)
    
    return Path(target_dir)


def rsync_to_remote(local_dir: Path, remote_base: str, dataset_id: str) -> None:
    """使用 rsync 将目录同步到远端。"""
    remote_path = f"{remote_base.rstrip('/')}/{dataset_id}/"
    cmd = [
        "rsync",
        "-avh",
        "--partial",
        "--progress",
        f"{local_dir}/",
        remote_path,
    ]
    print(f"[rsync] {local_dir} -> {remote_path}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 lerobot 数据集到本地")
    parser.add_argument(
        "--remote",
        default=None,
        help="可选：远端 rsync 目标，如 user@host:/path，指定后会将下载的数据同步到远端",
    )
    parser.add_argument(
        "--work-dir",
        default="/public/lerobot_downloads",
        help="本地下载目录",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="同步后保留本地文件，不清理（仅在指定 --remote 时有效）",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="可选，仅下载前 N 个数据集用于测试",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace API 令牌（用于认证和提高速率限制），可通过环境变量 HF_TOKEN 提供",
    )
    args = parser.parse_args()

    # 使用 HF_TOKEN 进行认证
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        print(f"使用 HF_TOKEN 登录...")
        login(token=hf_token)

    api = HfApi()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset_ids = fetch_lerobot_dataset_ids(api)
    except Exception as exc:
        print(f"获取数据集列表失败: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.max:
        dataset_ids = dataset_ids[: args.max]

    print(f"即将下载 {len(dataset_ids)} 个数据集")
    print(f"下载目录: {work_dir}")
    if args.remote:
        print(f"完成后将同步到: {args.remote}")

    success_count = 0
    failed_count = 0
    failed_datasets = []

    for i, ds_id in enumerate(dataset_ids, 1):
        print(f"\n[{i}/{len(dataset_ids)}] {ds_id}")
        start = time.time()
        try:
            local_path = download_one(ds_id, work_dir)
            
            if args.remote:
                rsync_to_remote(local_path, args.remote, ds_id)
            
            success_count += 1
        except subprocess.CalledProcessError as exc:
            print(f"  rsync 失败: {exc}", file=sys.stderr)
            failed_count += 1
            failed_datasets.append((ds_id, str(exc)))
        except Exception as exc:  # noqa: BLE001
            print(f"  下载失败: {exc}", file=sys.stderr)
            failed_count += 1
            failed_datasets.append((ds_id, str(exc)))
        else:
            duration = time.time() - start
            print(f"  完成，用时 {duration/60:.2f} 分钟")
            if args.remote and not args.keep_local:
                try:
                    shutil.rmtree(local_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"  清理本地失败: {exc}", file=sys.stderr)
        
        # 请求间隔，避免被限流
        time.sleep(2)

    print("\n" + "="*80)
    print(f"完成统计: 成功 {success_count}, 失败 {failed_count}")
    if failed_datasets:
        print("\n失败的数据集:")
        for ds_id, error in failed_datasets:
            print(f"  - {ds_id}: {error}")
    print("="*80)


if __name__ == "__main__":
    main()
