import pandas as pd
import os

excel_path = 'results/experiment_results.xlsx'

if not os.path.exists(excel_path):
    print(f"Excel文件不存在: {excel_path}")
    exit(1)

df = pd.read_excel(excel_path)

print("="*70)
print("Excel详细检查")
print("="*70)

print(f"\n总记录数: {len(df)}")
print(f"列名: {list(df.columns)}")

print("\n最新20条记录:")
print(df[['Method', 'Dataset', 'Learning_Rate', 'Accuracy', 'Notes']].tail(20).to_string(index=False))

print("\n\nLearning_Rate详细统计:")
print(df['Learning_Rate'].describe())
print(f"\n唯一值: {df['Learning_Rate'].unique()}")
print(f"数据类型: {df['Learning_Rate'].dtype}")

# 检查0值
zero_lr = df[df['Learning_Rate'] == 0]
print(f"\nLearning_Rate=0的记录数: {len(zero_lr)}")
if len(zero_lr) > 0:
    print(zero_lr[['Method', 'Dataset', 'Learning_Rate']].to_string())

# 检查错误
print("\n\n错误记录:")
if 'Notes' in df.columns:
    errors = df[df['Notes'].notna() & df['Notes'].str.contains('ERROR', na=False)]
    print(f"错误数: {len(errors)}")
    if len(errors) > 0:
        print(errors[['Method', 'Dataset', 'Learning_Rate', 'Accuracy', 'Notes']].to_string(index=False))
else:
    print("没有Notes列")

# 检查每个方法的Learning_Rate
print("\n\n各方法的Learning_Rate:")
for method in sorted(df['Method'].unique()):
    method_data = df[df['Method'] == method]
    lr_values = method_data['Learning_Rate'].unique()
    print(f"{method:12s}: {lr_values}")
