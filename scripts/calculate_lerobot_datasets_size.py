#!/usr/bin/env python3
"""
计算 HuggingFace 上所有 label 为 lerobot 的数据集的总大小。
"""

from huggingface_hub import HfApi
from typing import List, Dict, Optional
import argparse
import time


def get_dataset_size_in_gb(size_bytes: int) -> float:
    """将字节转换为 GB."""
    return size_bytes / (1024 ** 3)


def get_dataset_size_from_repo(api: HfApi, dataset_id: str) -> Optional[int]:
    """
    从 HuggingFace dataset_info 获取数据集的实际大小（字节）。
    
    Args:
        api: HfApi 实例
        dataset_id: 数据集 ID (e.g., "lerobot/aloha_sim_insertion_human")
        
    Returns:
        大小（字节）或 None
    """
    try:
        info = api.dataset_info(repo_id=dataset_id)
        # 使用 usedStorage 属性获取总大小
        if hasattr(info, 'usedStorage') and info.usedStorage is not None:
            return info.usedStorage
        return None
    except Exception as e:
        print(f"  获取 {dataset_id} 大小失败: {e}")
        return None


def fetch_lerobot_datasets() -> List[Dict]:
    """
    从 HuggingFace Hub 获取所有 label 为 lerobot 的数据集。
    
    Returns:
        List[Dict]: 数据集列表，包含名称和大小信息
    """
    lerobot_datasets = []
    api = HfApi()
    
    print("正在查询 HuggingFace Hub 上标签为 'lerobot' 的数据集...")
    
    try:
        # 使用 HfApi.list_datasets 函数获取数据集
        datasets_generator = api.list_datasets(author="lerobot", full=True)
        
        for i, dataset in enumerate(datasets_generator):
            dataset_id = dataset.id
            print(f"[{i+1}] 处理数据集: {dataset_id}")
            
            # 获取大小信息
            size_bytes = get_dataset_size_from_repo(api, dataset_id)
            
            dataset_info = {
                "id": dataset_id,
                "name": dataset_id.split("/")[-1],
                "author": dataset_id.split("/")[0],
                "size_bytes": size_bytes,
                "created_at": dataset.created_at if hasattr(dataset, 'created_at') else None,
                "downloads": dataset.downloads if hasattr(dataset, 'downloads') else 0,
                "likes": dataset.likes if hasattr(dataset, 'likes') else 0,
            }
            lerobot_datasets.append(dataset_info)
            
            # 添加延迟以避免 API 限流
            time.sleep(0.1)
    
    except Exception as e:
        print(f"错误: {e}")
    
    return lerobot_datasets


def calculate_total_size(datasets: List[Dict]) -> None:
    """
    计算并显示数据集的总大小。
    
    Args:
        datasets: 数据集列表
    """
    if not datasets:
        print("未找到任何标签为 'lerobot' 的数据集。")
        return
    
    print("\n" + "="*80)
    print(f"找到 {len(datasets)} 个数据集")
    print("="*80 + "\n")
    
    total_size_bytes = 0
    datasets_with_size = []
    datasets_without_size = []
    
    for dataset in datasets:
        if dataset["size_bytes"] is not None:
            datasets_with_size.append(dataset)
            total_size_bytes += dataset["size_bytes"]
        else:
            datasets_without_size.append(dataset)
    
    # 显示详细信息
    print(f"{'数据集 ID':<50} {'大小 (GB)':<15} {'下载次数':<10}")
    print("-" * 80)
    
    for dataset in sorted(datasets_with_size, key=lambda x: x["size_bytes"], reverse=True):
        size_gb = get_dataset_size_in_gb(dataset["size_bytes"])
        print(f"{dataset['id']:<50} {size_gb:>13.2f} GB  {dataset['downloads']:>8}")
    
    print("-" * 80)
    print(f"\n包含大小信息的数据集: {len(datasets_with_size)}")
    print(f"缺少大小信息的数据集: {len(datasets_without_size)}")
    
    if datasets_without_size:
        print("\n缺少大小信息的数据集:")
        for dataset in datasets_without_size:
            print(f"  - {dataset['id']}")
    
    # 显示统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    total_size_gb = get_dataset_size_in_gb(total_size_bytes)
    total_size_tb = total_size_gb / 1024
    
    print(f"总大小: {total_size_bytes:,} 字节")
    print(f"总大小: {total_size_gb:,.2f} GB")
    print(f"总大小: {total_size_tb:,.4f} TB")
    
    if datasets_with_size:
        avg_size_gb = get_dataset_size_in_gb(total_size_bytes / len(datasets_with_size))
        print(f"平均大小: {avg_size_gb:,.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="计算 HuggingFace 上所有标签为 lerobot 的数据集的总大小"
    )
    args = parser.parse_args()
    
    # 获取数据集
    datasets = fetch_lerobot_datasets()
    
    # 计算和显示统计信息
    calculate_total_size(datasets)


if __name__ == "__main__":
    main()
