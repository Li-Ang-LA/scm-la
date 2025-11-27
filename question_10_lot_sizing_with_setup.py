"""
VeloMotion GmbH - 供应链优化
第十问：带设置成本和最小批量的生产计划模型

问题描述：
- 设置成本（Setup Cost）：€5,000/次（每当决定生产时产生）
- 最小批量约束（Minimum Batch Size）：50单位（如果生产，至少生产50单位）
- 这是一个经典的生产批量问题（Lot Sizing Problem）

数学模型方法：
1. 引入二进制决策变量 Y[t] 表示第t月是否进行生产
2. 如果Y[t]=1，生产量 P[t] >= 50
3. 如果Y[t]=0，生产量 P[t] = 0（不生产）
4. 目标函数中加入设置成本：setup_cost * Y[t]
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# ============================================================================
# 1. 数据定义
# ============================================================================

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
demand = [2800, 3200, 4000, 4500, 5500, 5200,
          4800, 4300, 4700, 5400, 6000, 6800]

num_months = len(months)

# 成本参数
wage_regular = 19              # 正常工资 €/小时
wage_overtime = 24             # 加班工资 €/小时
hire_cost = 950                # 招聘成本 €/员工
layoff_cost = 1600             # 裁员成本 €/员工
inventory_cost = 12            # 库存成本 €/单位/月
backorder_cost = 35            # 积压订单成本 €/单位/月
material_cost = 620            # 原材料和零件成本 €/单位
outsource_cost = 710           # 外部组装成本 €/单位
revenue_price = 1250           # 销售价格 €/单位

# ============================================================================
# 2. 新增参数：设置成本和最小批量
# ============================================================================

setup_cost = 5000              # 设置成本 €/次（每月首次生产产生）
min_batch_size = 50            # 最小批量 单位（如果生产，至少生产50单位）
max_production_per_month = 10000  # 为了建模需要的上界（一个足够大的数）

# 生产参数
labor_hours_per_unit = 3.5     # 每单位所需工时
days_per_month = 22            # 每月工作天数
hours_per_shift = 8            # 每班小时数
max_overtime_per_worker = 12   # 每员工每月最大加班小时数
initial_workers = 160          # 初始员工数
initial_inventory = 600        # 初始库存
max_hire_per_month = 50        # 每月最大招聘数
max_inventory_capacity = 8000  # 最大库存容量

# 计算每员工每月可用工时
hours_per_worker_per_month = days_per_month * hours_per_shift  # 176小时

print("="*80)
print("VeloMotion GmbH - 第十问：带设置成本和最小批量的生产计划")
print("="*80)
print()

print("模型参数：")
print(f"  设置成本：€{setup_cost:,}/次（每当开始生产时产生）")
print(f"  最小批量：{min_batch_size}单位（如果生产，至少生产此数量）")
print(f"  基础模型库存成本：€{inventory_cost}/单位/月")
print()

# ============================================================================
# 3. 创建Gurobi模型
# ============================================================================

m = gp.Model("VeloMotion_Q10_LotSizing")
m.setParam('OutputFlag', 1)

# ============================================================================
# 4. 决策变量
# ============================================================================

# 内部生产量 (单位)
P = m.addVars(num_months, name="Production", lb=0)

# 外部组装量 (单位)
O = m.addVars(num_months, name="Outsource", lb=0)

# 员工数量 (人) - 整数
W = m.addVars(num_months, name="Workers", vtype=GRB.INTEGER, lb=0)

# 招聘数量 (人) - 整数
H = m.addVars(num_months, name="Hire", vtype=GRB.INTEGER, lb=0)

# 裁员数量 (人) - 整数
L = m.addVars(num_months, name="Layoff", vtype=GRB.INTEGER, lb=0)

# 加班小时数 (小时)
OT = m.addVars(num_months, name="Overtime", lb=0)

# 库存量 (单位)
I = m.addVars(num_months, name="Inventory", lb=0)

# 积压订单量 (单位)
B = m.addVars(num_months, name="Backorder", lb=0)

# ============================================================================
# 5. 新增决策变量：生产指示变量
# ============================================================================

# 二进制变量：第t月是否进行内部生产
# Y[t] = 1 表示第t月进行内部生产（需要支付设置成本）
# Y[t] = 0 表示第t月不进行内部生产
Y = m.addVars(num_months, name="ProductionSetup", vtype=GRB.BINARY)

m.update()

# ============================================================================
# 6. 约束条件
# ============================================================================

# 6.1 库存平衡约束
# 上期库存 + 本期生产 + 外部组装 + 上期积压 = 本期需求 + 本期库存 + 本期积压
for t in range(num_months):
    supply_prev = initial_inventory if t == 0 else I[t-1]
    backorder_prev = 0 if t == 0 else B[t-1]

    m.addConstr(
        supply_prev + P[t] + O[t] + backorder_prev == demand[t] + I[t] + B[t],
        name=f"InventoryBalance_t{t+1}"
    )

# 6.2 员工数量平衡约束
# W[t] = W[t-1] + H[t] - L[t]
for t in range(num_months):
    workers_prev = initial_workers if t == 0 else W[t-1]
    m.addConstr(
        W[t] == workers_prev + H[t] - L[t],
        name=f"WorkerBalance_t{t+1}"
    )

# 6.3 招聘限制
for t in range(num_months):
    m.addConstr(
        H[t] <= max_hire_per_month,
        name=f"HireLimit_t{t+1}"
    )

# 6.4 库存容量限制
for t in range(num_months):
    m.addConstr(
        I[t] <= max_inventory_capacity,
        name=f"CapacityLimit_t{t+1}"
    )

# 6.5 加班时间限制
for t in range(num_months):
    m.addConstr(
        OT[t] <= W[t] * max_overtime_per_worker,
        name=f"OvertimeLimit_t{t+1}"
    )

# 6.6 生产所需工时约束
# 正常工时 + 加班时间 >= 生产所需工时
for t in range(num_months):
    m.addConstr(
        W[t] * hours_per_worker_per_month + OT[t] >= P[t] * labor_hours_per_unit,
        name=f"ProductionCapacity_t{t+1}"
    )

# 6.7 期末必须清零积压订单
m.addConstr(B[num_months - 1] == 0, name="EndNoBackorder")

# ============================================================================
# 7. 新增约束：设置成本和最小批量约束
# ============================================================================

print("新增约束条件：")
print()

# 7.1 最小批量约束
# 如果Y[t]=1（生产），则 P[t] >= min_batch_size
# 如果Y[t]=0（不生产），则 P[t] = 0
# 数学表达：P[t] >= min_batch_size * Y[t]
for t in range(num_months):
    m.addConstr(
        P[t] >= min_batch_size * Y[t],
        name=f"MinBatchSize_t{t+1}"
    )

# 7.2 生产上界约束
# P[t] <= max_production_per_month * Y[t]
# 确保如果Y[t]=0，则P[t]=0；如果Y[t]=1，则P[t]可以最多为上界值
for t in range(num_months):
    m.addConstr(
        P[t] <= max_production_per_month * Y[t],
        name=f"ProductionUpperBound_t{t+1}"
    )

print(f"✓ 最小批量约束：P[t] >= {min_batch_size} × Y[t]")
print(f"  （如果生产，至少生产{min_batch_size}单位）")
print()
print(f"✓ 生产上界约束：P[t] <= {max_production_per_month} × Y[t]")
print(f"  （强制Y[t]与P[t]的关系）")
print()

m.update()

# ============================================================================
# 8. 目标函数 - 最大化利润（包含设置成本）
# ============================================================================

# 收入
total_revenue = gp.quicksum((P[t] + O[t]) * revenue_price for t in range(num_months))

# 各项成本
material_costs = gp.quicksum(P[t] * material_cost for t in range(num_months))
outsource_costs = gp.quicksum(O[t] * outsource_cost for t in range(num_months))
labor_regular_costs = gp.quicksum(P[t] * labor_hours_per_unit * wage_regular for t in range(num_months))
labor_overtime_costs = gp.quicksum(OT[t] * wage_overtime for t in range(num_months))
hire_costs = gp.quicksum(H[t] * hire_cost for t in range(num_months))
layoff_costs = gp.quicksum(L[t] * layoff_cost for t in range(num_months))
inventory_costs = gp.quicksum(I[t] * inventory_cost for t in range(num_months))
backorder_costs = gp.quicksum(B[t] * backorder_cost for t in range(num_months))

# ============================================================================
# 新增成本：设置成本
# ============================================================================
setup_costs = gp.quicksum(setup_cost * Y[t] for t in range(num_months))

total_costs = (material_costs + outsource_costs + labor_regular_costs +
               labor_overtime_costs + hire_costs + layoff_costs +
               inventory_costs + backorder_costs + setup_costs)

# 最大化利润
profit = total_revenue - total_costs
m.setObjective(profit, GRB.MAXIMIZE)

m.update()

# ============================================================================
# 9. 求解
# ============================================================================

print("="*80)
print("求解包含设置成本和最小批量的生产计划模型...")
print("="*80)
print()

m.optimize()

# ============================================================================
# 10. 输出结果
# ============================================================================

if m.status == GRB.OPTIMAL:
    print("\n✓ 找到最优解！")
    print()

    # 获取数值
    total_revenue_val = total_revenue.getValue()
    material_cost_val = material_costs.getValue()
    outsource_cost_val = outsource_costs.getValue()
    labor_regular_val = labor_regular_costs.getValue()
    labor_overtime_val = labor_overtime_costs.getValue()
    hire_cost_val = hire_costs.getValue()
    layoff_cost_val = layoff_costs.getValue()
    inventory_cost_val = inventory_costs.getValue()
    backorder_cost_val = backorder_costs.getValue()
    setup_cost_val = setup_costs.getValue()
    total_cost_val = total_costs.getValue()
    profit_val = profit.getValue()

    # 输出第十问的答案
    print("="*80)
    print("问题10：带设置成本和最小批量的生产计划")
    print("="*80)
    print()

    print(f"年度最优利润: €{profit_val:,.2f}")
    print()

    print("成本结构分析:")
    print("-" * 80)
    print(f"{'成本项目':<30} {'金额(€)':<20} {'占比(%)':<15}")
    print("-" * 80)

    cost_items = [
        ("原材料和零件成本", material_cost_val),
        ("外部组装成本", outsource_cost_val),
        ("正常工资成本", labor_regular_val),
        ("加班工资成本", labor_overtime_val),
        ("招聘成本", hire_cost_val),
        ("裁员成本", layoff_cost_val),
        ("库存成本", inventory_cost_val),
        ("积压订单成本", backorder_cost_val),
        ("设置成本", setup_cost_val),  # 新增
    ]

    for item_name, item_val in cost_items:
        pct = (item_val / total_cost_val * 100) if total_cost_val > 0 else 0
        print(f"{item_name:<30} €{item_val:>18,.2f} {pct:>13.2f}%")

    print("-" * 80)
    print(f"{'总成本':<30} €{total_cost_val:>18,.2f} {100.00:>13.2f}%")
    print()
    print(f"总收入: €{total_revenue_val:,.2f}")
    print()

    # 提取生产指示变量的值
    production_setup_months = []
    for t in range(num_months):
        if Y[t].X > 0.5:
            production_setup_months.append(t)

    print("生产决策分析:")
    print("-" * 80)
    print(f"进行生产的月份数：{len(production_setup_months)}")
    print(f"进行生产的具体月份：{[months[t] for t in production_setup_months]}")
    print(f"设置成本总额：€{setup_cost_val:,.2f} ({len(production_setup_months)}个月 × €{setup_cost:,}/月)")
    print()

    # 月度详细表
    print("="*80)
    print("月度生产计划与设置决策")
    print("="*80)
    print()

    results_data = []
    for t in range(num_months):
        results_data.append({
            'Month': months[t],
            'Demand': int(demand[t]),
            'Production': int(P[t].X),
            'Setup': 'Yes' if Y[t].X > 0.5 else 'No',
            'Outsource': int(O[t].X),
            'Workers': int(W[t].X),
            'Inventory': int(I[t].X),
            'Backorder': int(B[t].X)
        })

    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    print()

    # 关键统计
    print("="*80)
    print("关键统计指标")
    print("="*80)
    print()
    total_production = sum(P[t].X for t in range(num_months))
    total_outsource = sum(O[t].X for t in range(num_months))
    total_demand = sum(demand)
    total_hire = sum(H[t].X for t in range(num_months))
    total_layoff = sum(L[t].X for t in range(num_months))
    total_overtime = sum(OT[t].X for t in range(num_months))
    num_setup = sum(Y[t].X for t in range(num_months))

    print(f"总需求量:          {int(total_demand):>10} 单位")
    print(f"内部生产:          {int(total_production):>10} 单位 ({total_production/total_demand*100:.1f}%)")
    print(f"外部组装:          {int(total_outsource):>10} 单位 ({total_outsource/total_demand*100:.1f}%)")
    print(f"总招聘数:          {int(total_hire):>10} 人")
    print(f"总裁员数:          {int(total_layoff):>10} 人")
    print(f"总加班小时:        {int(total_overtime):>10} 小时")
    print(f"生产批次数:        {int(num_setup):>10} 次（触发设置成本）")
    print()

    # ========================================================================
    # 11. 关键分析：与基础模型的对比
    # ========================================================================

    print("="*80)
    print("关键发现：最小批量和设置成本的影响")
    print("="*80)
    print()

    print("1. 生产批次和设置成本")
    print("-" * 80)
    print(f"   • 生产批次总数：{int(num_setup)}次")
    print(f"   • 总设置成本：€{setup_cost_val:,.2f}")
    print(f"   • 平均每次设置成本：€{setup_cost_val/max(num_setup, 1):,.2f}")
    print()

    print("2. 最小批量约束的影响")
    print("-" * 80)
    # 分析每个生产月份的批量
    batch_analysis = []
    for t in production_setup_months:
        batch_analysis.append({
            'Month': months[t],
            'Batch_Size': int(P[t].X),
            'Min_Required': min_batch_size,
            'Above_Min': int(P[t].X) - min_batch_size
        })

    if batch_analysis:
        df_batch = pd.DataFrame(batch_analysis)
        print(df_batch.to_string(index=False))
        print()
        print(f"   • 所有批量都满足最小批量约束（>= {min_batch_size}单位）")
        avg_batch = sum(int(P[t].X) for t in production_setup_months) / max(len(production_setup_months), 1)
        print(f"   • 平均批量：{int(avg_batch)}单位")
    print()

    print("3. 设置成本与库存成本的权衡")
    print("-" * 80)
    print(f"   • 总库存成本：€{inventory_cost_val:,.2f}")
    print(f"   • 总设置成本：€{setup_cost_val:,.2f}")
    print(f"   • 成本比例：库存/设置 = {inventory_cost_val/max(setup_cost_val, 1):.2f}")
    print()
    if setup_cost_val > 0:
        print("   解释：")
        if inventory_cost_val > setup_cost_val:
            print("   → 库存成本较高，优化器倾向于减少库存，增加生产频次")
        else:
            print("   → 设置成本较高，优化器倾向于减少生产频次，增加库存")
    print()

    print("4. 生产策略分析")
    print("-" * 80)
    print(f"   • 有生产的月份：{len(production_setup_months)}个")
    print(f"   • 无生产的月份：{num_months - len(production_setup_months)}个")
    print(f"   • 生产集中度：{len(production_setup_months)/num_months*100:.1f}%")
    print()

    if len(production_setup_months) < num_months:
        no_production_months = [months[t] for t in range(num_months) if Y[t].X < 0.5]
        print(f"   无生产的月份（从库存供应）：{', '.join(no_production_months)}")
        print()

    # ========================================================================
    # 12. 数学建模的设计原理解释
    # ========================================================================

    print("="*80)
    print("数学建模设计原理")
    print("="*80)
    print()

    print("约束条件设计：")
    print()
    print("1. 最小批量约束（Big-M 形式）")
    print("   P[t] >= min_batch_size × Y[t]")
    print("   • Y[t]=1 时：P[t] >= 50（强制最小批量）")
    print("   • Y[t]=0 时：P[t] >= 0（不强制，但下一个约束使其=0）")
    print()

    print("2. 生产上界约束（强制 Y[t] 与 P[t] 的关系）")
    print("   P[t] <= 10000 × Y[t]")
    print("   • Y[t]=1 时：P[t] 可以最多为10000")
    print("   • Y[t]=0 时：P[t] <= 0，结合约束1，得P[t]=0")
    print()

    print("3. 目标函数包含设置成本")
    print("   setup_cost × Y[t]：每次生产（Y[t]=1）支付€5,000")
    print()

    print("模型类型：")
    print("   • Mixed Integer Linear Programming (MILP)")
    print("   • 包含：连续变量、整数变量、二进制变量")
    print()

else:
    print(f"求解失败！状态码: {m.status}")

print("\n" + "="*80)
