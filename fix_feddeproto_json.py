"""
修复已有的FedDeProto JSON文件
将负数loss替换为正确的Stage2分类loss
"""
import json
from pathlib import Path

def fix_feddeproto_json():
    """修复FedDeProto的JSON文件"""
    logs_dir = Path('results/logs')
    
    if not logs_dir.exists():
        print("❌ logs目录不存在")
        return
    
    # 查找所有FedDeProto的JSON文件
    feddeproto_files = list(logs_dir.glob('*_feddeproto.json'))
    
    if not feddeproto_files:
        print("❌ 未找到FedDeProto的JSON文件")
        return
    
    print(f"找到 {len(feddeproto_files)} 个FedDeProto JSON文件")
    print("="*80)
    
    for json_file in feddeproto_files:
        print(f"\n处理: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有负数loss
            loss_history = data['history']['loss']
            has_negative = any(x < 0 for x in loss_history)
            
            if not has_negative:
                print("  ✅ 没有负数loss，无需修复")
                continue
            
            print(f"  ⚠️ 检测到负数loss")
            print(f"     总轮次: {len(loss_history)}")
            print(f"     第一个loss: {loss_history[0]:.4f}")
            print(f"     最后一个loss: {loss_history[-1]:.4f}")
            
            # 分析：通常前100轮是Stage1（可能为负），后150轮是Stage2（正数）
            # 我们需要找到Stage1和Stage2的分界点
            
            # 方法1: 如果有stage1_rounds信息
            if 'feddeproto_info' in data:
                stage1_rounds = data['feddeproto_info']['stage1_rounds']
                stage2_rounds = data['feddeproto_info']['stage2_rounds']
            else:
                # 方法2: 寻找loss从负转正的点，或者假设前100轮是Stage1
                stage1_rounds = 100
                stage2_rounds = len(loss_history) - stage1_rounds
            
            print(f"     Stage1轮次: {stage1_rounds}")
            print(f"     Stage2轮次: {stage2_rounds}")
            
            # 分离Stage1和Stage2的loss
            stage1_loss = loss_history[:stage1_rounds]
            stage2_loss = loss_history[stage1_rounds:]
            
            if not stage2_loss:
                print("  ❌ 无法找到Stage2 loss")
                continue
            
            # 更新数据结构
            data['history']['loss'] = stage2_loss  # 主loss只保留Stage2
            data['final_loss'] = float(stage2_loss[-1])  # 最终loss
            
            # 添加详细信息
            if 'feddeproto_info' not in data:
                data['feddeproto_info'] = {}
            
            data['feddeproto_info']['stage1_rounds'] = stage1_rounds
            data['feddeproto_info']['stage2_rounds'] = stage2_rounds
            data['feddeproto_info']['stage1_loss'] = stage1_loss
            data['feddeproto_info']['stage2_loss'] = stage2_loss
            data['feddeproto_info']['note'] = 'stage1_loss来自VAE-WGAN-GP可能为负，stage2_loss为分类loss必为正'
            
            # 保存修复后的文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"  ✅ 修复完成")
            print(f"     新的loss范围: {min(stage2_loss):.4f} ~ {max(stage2_loss):.4f}")
            print(f"     最终loss: {stage2_loss[-1]:.4f}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n" + "="*80)
    print("修复完成！")

if __name__ == '__main__':
    fix_feddeproto_json()
