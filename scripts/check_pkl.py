#!/usr/bin/env python3
"""
检查 HIL 收集的 pkl 数据文件

用法:
    python scripts/check_pkl.py <pkl文件路径>
    python scripts/check_pkl.py ./collected_data/intervention_20260128_172237.pkl
"""
import sys
import pickle
import numpy as np
from pathlib import Path


def check_pkl(filepath: str, num_samples: int = 5):
    """检查 pkl 文件内容"""
    
    path = Path(filepath)
    if not path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"📁 文件: {path.name}")
    print(f"   路径: {path.absolute()}")
    print(f"   大小: {path.stat().st_size / 1024:.1f} KB")
    print('='*60)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n📊 数据概览")
    print(f"   类型: {type(data).__name__}")
    print(f"   数量: {len(data)} 条 transitions")
    
    if len(data) == 0:
        print("   ⚠️ 数据为空")
        return
    
    # 检查第一条数据的结构
    sample = data[0]
    print(f"\n📋 数据字段:")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"   {k}: np.ndarray, shape={v.shape}, dtype={v.dtype}")
            if v.size < 50:  # 小数组直接显示
                print(f"      值: {v.flatten()[:10]}{'...' if v.size > 10 else ''}")
            else:
                print(f"      范围: [{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}")
        elif isinstance(v, dict):
            print(f"   {k}: dict with {len(v)} keys")
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    print(f"      {kk}: shape={vv.shape}, range=[{vv.min():.3f}, {vv.max():.3f}]")
                elif hasattr(vv, 'shape'):  # torch.Tensor
                    print(f"      {kk}: shape={vv.shape}, type={type(vv).__name__}")
                else:
                    print(f"      {kk}: {type(vv).__name__}")
        elif isinstance(v, (int, float, bool)):
            print(f"   {k}: {v}")
        elif isinstance(v, str):
            print(f"   {k}: '{v}'")
        else:
            print(f"   {k}: {type(v).__name__}")
    
    # 显示动作样例
    print(f"\n🎯 动作样例 (前 {min(num_samples, len(data))} 条):")
    for i in range(min(num_samples, len(data))):
        trans = data[i]
        action = trans.get('actions', trans.get('action'))
        if action is not None:
            if isinstance(action, np.ndarray):
                # 格式化显示
                arm = action[:7] if len(action) >= 7 else action
                grip = action[7] if len(action) > 7 else None
                waist = action[8:11] if len(action) >= 11 else None
                head = action[11:13] if len(action) >= 13 else None
                
                line = f"  [{i:3d}] arm={np.round(arm, 3)}"
                if grip is not None:
                    line += f", grip={grip:.3f}"
                if waist is not None:
                    line += f", waist={np.round(waist, 3)}"
                if head is not None:
                    line += f", head={np.round(head, 3)}"
                print(line)
            else:
                print(f"  [{i:3d}] action: {action}")
    
    # 统计信息
    print(f"\n📈 统计信息:")
    
    # 干预统计
    interventions = [t.get('is_intervention', False) for t in data]
    intervention_count = sum(interventions)
    print(f"   干预比例: {intervention_count}/{len(data)} ({100*intervention_count/len(data):.1f}%)")
    
    # 奖励统计
    rewards = [t.get('rewards', t.get('reward', 0)) for t in data]
    if any(r != 0 for r in rewards):
        print(f"   奖励: mean={np.mean(rewards):.4f}, sum={np.sum(rewards):.4f}")
    
    # 来源统计
    sources = [t.get('source', 'unknown') for t in data]
    source_counts = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1
    print(f"   来源分布: {source_counts}")
    
    print(f"\n{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        
        # 尝试列出 collected_data 目录
        collected_dir = Path("./collected_data")
        if collected_dir.exists():
            files = list(collected_dir.glob("*.pkl"))
            if files:
                print(f"\n找到 {len(files)} 个 pkl 文件:")
                for f in sorted(files)[-10:]:  # 显示最近10个
                    print(f"  {f}")
                print(f"\n示例: python scripts/check_pkl.py {files[-1]}")
        return
    
    for filepath in sys.argv[1:]:
        check_pkl(filepath)


if __name__ == "__main__":
    main()
