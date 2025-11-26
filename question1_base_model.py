"""
VeloMotion GmbH - 供应链优化
第一问：基础生产计划模型
求解：年度利润 + 成本结构分析
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

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

# ============================================================================
# 2. 创建Gurobi模型
# ============================================================================

m = gp.Model("VeloMotion_Q1_BaseModel")
m.setParam('OutputFlag', 1)

# ============================================================================
# 3. 决策变量
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

m.update()

# ============================================================================
# 4. 约束条件
# ============================================================================

# 4.1 库存平衡约束
# 上期库存 + 本期生产 + 外部组装 + 上期积压 = 本期需求 + 本期库存 + 本期积压
for t in range(num_months):
    supply_prev = initial_inventory if t == 0 else I[t-1]
    backorder_prev = 0 if t == 0 else B[t-1]

    m.addConstr(
        supply_prev + P[t] + O[t] + backorder_prev == demand[t] + I[t] + B[t],
        name=f"InventoryBalance_t{t+1}"
    )

# 4.2 员工数量平衡约束
# W[t] = W[t-1] + H[t] - L[t]
for t in range(num_months):
    workers_prev = initial_workers if t == 0 else W[t-1]
    m.addConstr(
        W[t] == workers_prev + H[t] - L[t],
        name=f"WorkerBalance_t{t+1}"
    )

# 4.3 招聘限制
for t in range(num_months):
    m.addConstr(
        H[t] <= max_hire_per_month,
        name=f"HireLimit_t{t+1}"
    )

# 4.4 库存容量限制
for t in range(num_months):
    m.addConstr(
        I[t] <= max_inventory_capacity,
        name=f"CapacityLimit_t{t+1}"
    )

# 4.5 加班时间限制
for t in range(num_months):
    m.addConstr(
        OT[t] <= W[t] * max_overtime_per_worker,
        name=f"OvertimeLimit_t{t+1}"
    )

# 4.6 生产所需工时约束
# 正常工时 + 加班时间 >= 生产所需工时
for t in range(num_months):
    m.addConstr(
        W[t] * hours_per_worker_per_month + OT[t] >= P[t] * labor_hours_per_unit,
        name=f"ProductionCapacity_t{t+1}"
    )

# 4.7 期末必须清零积压订单
m.addConstr(B[num_months - 1] == 0, name="EndNoBackorder")

m.update()

# ============================================================================
# 5. 目标函数 - 最大化利润
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

total_costs = (material_costs + outsource_costs + labor_regular_costs +
               labor_overtime_costs + hire_costs + layoff_costs +
               inventory_costs + backorder_costs)

# 最大化利润
profit = total_revenue - total_costs
m.setObjective(profit, GRB.MAXIMIZE)

m.update()

# ============================================================================
# 6. 求解
# ============================================================================

print("\n" + "="*80)
print("VeloMotion GmbH - 第一问：基础生产计划模型")
print("="*80)
print()

m.optimize()

# ============================================================================
# 7. 输出结果
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
    total_cost_val = total_costs.getValue()
    profit_val = profit.getValue()

    # 输出第一问的答案
    print("="*80)
    print("问题1：年度利润 + 成本结构")
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
    ]

    for item_name, item_val in cost_items:
        pct = (item_val / total_cost_val * 100) if total_cost_val > 0 else 0
        print(f"{item_name:<30} €{item_val:>18,.2f} {pct:>13.2f}%")

    print("-" * 80)
    print(f"{'总成本':<30} €{total_cost_val:>18,.2f} {100.00:>13.2f}%")
    print()
    print(f"总收入: €{total_revenue_val:,.2f}")
    print()

    # 月度详细表
    print("="*80)
    print("月度生产计划与成本分解")
    print("="*80)
    print()

    results_data = []
    for t in range(num_months):
        results_data.append({
            'Month': months[t],
            'Demand': int(demand[t]),
            'Production': int(P[t].X),
            'Outsource': int(O[t].X),
            'Workers': int(W[t].X),
            'Hire': int(H[t].X),
            'Layoff': int(L[t].X),
            'Overtime(h)': int(OT[t].X),
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

    print(f"总需求量:          {int(total_demand):>10} 单位")
    print(f"内部生产:          {int(total_production):>10} 单位 ({total_production/total_demand*100:.1f}%)")
    print(f"外部组装:          {int(total_outsource):>10} 单位 ({total_outsource/total_demand*100:.1f}%)")
    print(f"总招聘数:          {int(total_hire):>10} 人")
    print(f"总裁员数:          {int(total_layoff):>10} 人")
    print(f"总加班小时:        {int(total_overtime):>10} 小时")
    print()

else:
    print(f"求解失败！状态码: {m.status}")
