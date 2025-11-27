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

# ============================================================================
# 8. 问题8：单一促销活动的最优时机
# ============================================================================
"""
问题8：市场研究表明，一个月的5%价格折扣会：
- 该月销售增加35%
- 促销月的需求增加：原需求 * 0.35
- 后续两个月各拉动需求的6%：后续月份需求 * 0.06

目标：找到最优的促销月份，最大化年度利润
"""

print("\n" + "="*80)
print("问题8：单一促销活动的最优时机")
print("="*80)
print()

# 促销影响系数
promotion_sales_increase = 0.35    # 促销月销售增加35%
demand_pull_forward = 0.06         # 后续月份拉动6%的需求
price_reduction_rate = 0.05        # 价格折扣5%

# 为每个可能的促销月份创建模型
promotion_results = {}

for promotion_month in range(num_months):
    print(f"测试促销月份：{months[promotion_month]}")
    print("-" * 80)

    # 创建新模型
    m_promo = gp.Model(f"VeloMotion_Q8_Promotion_Month{promotion_month}")
    m_promo.setParam('OutputFlag', 0)  # 抑制输出

    # 决策变量（与基础模型相同）
    P_promo = m_promo.addVars(num_months, name="Production", lb=0)
    O_promo = m_promo.addVars(num_months, name="Outsource", lb=0)
    W_promo = m_promo.addVars(num_months, name="Workers", vtype=GRB.INTEGER, lb=0)
    H_promo = m_promo.addVars(num_months, name="Hire", vtype=GRB.INTEGER, lb=0)
    L_promo = m_promo.addVars(num_months, name="Layoff", vtype=GRB.INTEGER, lb=0)
    OT_promo = m_promo.addVars(num_months, name="Overtime", lb=0)
    I_promo = m_promo.addVars(num_months, name="Inventory", lb=0)
    B_promo = m_promo.addVars(num_months, name="Backorder", lb=0)

    m_promo.update()

    # 计算调整后的需求（考虑促销影响）
    adjusted_demand = []
    for t in range(num_months):
        if t == promotion_month:
            # 促销月：基础需求 + 基础需求*增长比例
            adjusted_demand.append(demand[t] * (1 + promotion_sales_increase))
        elif t > promotion_month and t <= promotion_month + 2:
            # 后续两个月：基础需求 - 拉动的需求
            adjusted_demand.append(demand[t] * (1 - demand_pull_forward))
        else:
            # 其他月份：保持不变
            adjusted_demand.append(demand[t])

    # 约束条件
    # 库存平衡约束
    for t in range(num_months):
        supply_prev = initial_inventory if t == 0 else I_promo[t-1]
        backorder_prev = 0 if t == 0 else B_promo[t-1]

        m_promo.addConstr(
            supply_prev + P_promo[t] + O_promo[t] + backorder_prev == adjusted_demand[t] + I_promo[t] + B_promo[t],
            name=f"InventoryBalance_t{t+1}"
        )

    # 员工数量平衡约束
    for t in range(num_months):
        workers_prev = initial_workers if t == 0 else W_promo[t-1]
        m_promo.addConstr(
            W_promo[t] == workers_prev + H_promo[t] - L_promo[t],
            name=f"WorkerBalance_t{t+1}"
        )

    # 招聘限制
    for t in range(num_months):
        m_promo.addConstr(H_promo[t] <= max_hire_per_month, name=f"HireLimit_t{t+1}")

    # 库存容量限制
    for t in range(num_months):
        m_promo.addConstr(I_promo[t] <= max_inventory_capacity, name=f"CapacityLimit_t{t+1}")

    # 加班时间限制
    for t in range(num_months):
        m_promo.addConstr(OT_promo[t] <= W_promo[t] * max_overtime_per_worker, name=f"OvertimeLimit_t{t+1}")

    # 生产所需工时约束
    for t in range(num_months):
        m_promo.addConstr(
            W_promo[t] * hours_per_worker_per_month + OT_promo[t] >= P_promo[t] * labor_hours_per_unit,
            name=f"ProductionCapacity_t{t+1}"
        )

    # 期末必须清零积压订单
    m_promo.addConstr(B_promo[num_months - 1] == 0, name="EndNoBackorder")

    m_promo.update()

    # 目标函数
    # 收入（考虑促销月的价格折扣）
    revenue_promo = 0
    for t in range(num_months):
        if t == promotion_month:
            # 促销月：价格 * (1 - 折扣率)
            price_t = revenue_price * (1 - price_reduction_rate)
        else:
            price_t = revenue_price
        revenue_promo += (P_promo[t] + O_promo[t]) * price_t

    # 成本
    material_costs_promo = gp.quicksum(P_promo[t] * material_cost for t in range(num_months))
    outsource_costs_promo = gp.quicksum(O_promo[t] * outsource_cost for t in range(num_months))
    labor_regular_costs_promo = gp.quicksum(P_promo[t] * labor_hours_per_unit * wage_regular for t in range(num_months))
    labor_overtime_costs_promo = gp.quicksum(OT_promo[t] * wage_overtime for t in range(num_months))
    hire_costs_promo = gp.quicksum(H_promo[t] * hire_cost for t in range(num_months))
    layoff_costs_promo = gp.quicksum(L_promo[t] * layoff_cost for t in range(num_months))
    inventory_costs_promo = gp.quicksum(I_promo[t] * inventory_cost for t in range(num_months))
    backorder_costs_promo = gp.quicksum(B_promo[t] * backorder_cost for t in range(num_months))

    total_costs_promo = (material_costs_promo + outsource_costs_promo + labor_regular_costs_promo +
                        labor_overtime_costs_promo + hire_costs_promo + layoff_costs_promo +
                        inventory_costs_promo + backorder_costs_promo)

    profit_promo = revenue_promo - total_costs_promo
    m_promo.setObjective(profit_promo, GRB.MAXIMIZE)

    m_promo.update()

    # 求解
    m_promo.optimize()

    if m_promo.status == GRB.OPTIMAL:
        promotion_results[promotion_month] = {
            'month': months[promotion_month],
            'profit': profit_promo.getValue(),
            'revenue': revenue_promo.getValue(),
            'total_cost': total_costs_promo.getValue(),
            'adjusted_demand': adjusted_demand,
            'production': [P_promo[t].X for t in range(num_months)],
            'outsource': [O_promo[t].X for t in range(num_months)],
            'workers': [W_promo[t].X for t in range(num_months)],
            'inventory': [I_promo[t].X for t in range(num_months)],
            'model': m_promo
        }
        print(f"  利润: €{profit_promo.getValue():,.2f}")
    else:
        print(f"  求解失败！状态码: {m_promo.status}")
    print()

# 找到最优促销月份
if promotion_results:
    best_promo_month = max(promotion_results.keys(),
                          key=lambda k: promotion_results[k]['profit'])
    best_result = promotion_results[best_promo_month]

    print("="*80)
    print("问题8：最优促销方案")
    print("="*80)
    print()
    print(f"最优促销月份: {best_result['month']}")
    print(f"促销年度利润: €{best_result['profit']:,.2f}")
    print(f"促销总收入: €{best_result['revenue']:,.2f}")
    print(f"促销总成本: €{best_result['total_cost']:,.2f}")
    print()

    # 与基础模型对比
    if m.status == GRB.OPTIMAL:
        profit_difference = best_result['profit'] - profit.getValue()
        print(f"与基础模型相比:")
        print(f"  利润差异: €{profit_difference:,.2f} ({profit_difference/profit.getValue()*100:+.2f}%)")
        print()

    # 促销月份的调整
    print(f"促销月份 {best_result['month']} 的需求调整:")
    print(f"  原始需求: {int(demand[best_promo_month])} 单位")
    print(f"  调整后需求: {int(best_result['adjusted_demand'][best_promo_month])} 单位")
    print(f"  增长: {int(best_result['adjusted_demand'][best_promo_month] - demand[best_promo_month])} 单位 ({(best_result['adjusted_demand'][best_promo_month] - demand[best_promo_month])/demand[best_promo_month]*100:.1f}%)")
    print()

# ============================================================================
# 9. 问题9：多个非连续促销活动的优化
# ============================================================================
"""
问题9：评估多个促销活动的效果（非连续月份）
假设：
- 促销不能在连续的月份进行
- 每次促销都应用相同的促销效果
- 目标：找到最优的促销数量和时机

采用枚举法测试所有可能的非连续促销组合
"""

print("\n" + "="*80)
print("问题9：多个非连续促销活动的优化")
print("="*80)
print()

from itertools import combinations

def is_non_consecutive(promotion_months):
    """检查促销月份是否为非连续"""
    if len(promotion_months) <= 1:
        return True
    sorted_months = sorted(promotion_months)
    for i in range(len(sorted_months) - 1):
        if sorted_months[i+1] - sorted_months[i] == 1:
            return False
    return True

# 存储所有有效的促销组合结果
multi_promotion_results = {}

# 测试1个、2个、3个等促销的所有可能组合
for num_promos in range(1, num_months + 1):
    print(f"测试 {num_promos} 个促销的所有可能组合...")

    valid_combinations = []
    for combo in combinations(range(num_months), num_promos):
        if is_non_consecutive(combo):
            valid_combinations.append(combo)

    print(f"  有效的非连续组合数: {len(valid_combinations)}")

    best_profit_for_num = None
    best_combo_for_num = None

    for combo in valid_combinations:
        # 创建多促销模型
        m_multi = gp.Model(f"VeloMotion_Q9_MultiPromo")
        m_multi.setParam('OutputFlag', 0)

        # 决策变量
        P_multi = m_multi.addVars(num_months, name="Production", lb=0)
        O_multi = m_multi.addVars(num_months, name="Outsource", lb=0)
        W_multi = m_multi.addVars(num_months, name="Workers", vtype=GRB.INTEGER, lb=0)
        H_multi = m_multi.addVars(num_months, name="Hire", vtype=GRB.INTEGER, lb=0)
        L_multi = m_multi.addVars(num_months, name="Layoff", vtype=GRB.INTEGER, lb=0)
        OT_multi = m_multi.addVars(num_months, name="Overtime", lb=0)
        I_multi = m_multi.addVars(num_months, name="Inventory", lb=0)
        B_multi = m_multi.addVars(num_months, name="Backorder", lb=0)

        m_multi.update()

        # 计算调整后的需求（考虑多个促销月份）
        adjusted_demand_multi = []
        for t in range(num_months):
            if t in combo:
                # 该月进行促销
                adjusted_demand_multi.append(demand[t] * (1 + promotion_sales_increase))
            elif any(t > promo_t and t <= promo_t + 2 for promo_t in combo):
                # 该月处于某个促销的影响期
                # 从所有相关的促销月份中汇总拉动
                total_pull_back = 0
                for promo_t in combo:
                    if promo_t < t <= promo_t + 2:
                        total_pull_back += demand[t] * demand_pull_forward
                # 避免负需求
                adjusted_demand_multi.append(max(demand[t] - total_pull_back, 0))
            else:
                adjusted_demand_multi.append(demand[t])

        # 约束条件（与问题8相同逻辑）
        for t in range(num_months):
            supply_prev = initial_inventory if t == 0 else I_multi[t-1]
            backorder_prev = 0 if t == 0 else B_multi[t-1]

            m_multi.addConstr(
                supply_prev + P_multi[t] + O_multi[t] + backorder_prev == adjusted_demand_multi[t] + I_multi[t] + B_multi[t],
                name=f"InventoryBalance_t{t+1}"
            )

        for t in range(num_months):
            workers_prev = initial_workers if t == 0 else W_multi[t-1]
            m_multi.addConstr(
                W_multi[t] == workers_prev + H_multi[t] - L_multi[t],
                name=f"WorkerBalance_t{t+1}"
            )

        for t in range(num_months):
            m_multi.addConstr(H_multi[t] <= max_hire_per_month, name=f"HireLimit_t{t+1}")

        for t in range(num_months):
            m_multi.addConstr(I_multi[t] <= max_inventory_capacity, name=f"CapacityLimit_t{t+1}")

        for t in range(num_months):
            m_multi.addConstr(OT_multi[t] <= W_multi[t] * max_overtime_per_worker, name=f"OvertimeLimit_t{t+1}")

        for t in range(num_months):
            m_multi.addConstr(
                W_multi[t] * hours_per_worker_per_month + OT_multi[t] >= P_multi[t] * labor_hours_per_unit,
                name=f"ProductionCapacity_t{t+1}"
            )

        m_multi.addConstr(B_multi[num_months - 1] == 0, name="EndNoBackorder")

        m_multi.update()

        # 目标函数
        revenue_multi = 0
        for t in range(num_months):
            if t in combo:
                price_t = revenue_price * (1 - price_reduction_rate)
            else:
                price_t = revenue_price
            revenue_multi += (P_multi[t] + O_multi[t]) * price_t

        material_costs_multi = gp.quicksum(P_multi[t] * material_cost for t in range(num_months))
        outsource_costs_multi = gp.quicksum(O_multi[t] * outsource_cost for t in range(num_months))
        labor_regular_costs_multi = gp.quicksum(P_multi[t] * labor_hours_per_unit * wage_regular for t in range(num_months))
        labor_overtime_costs_multi = gp.quicksum(OT_multi[t] * wage_overtime for t in range(num_months))
        hire_costs_multi = gp.quicksum(H_multi[t] * hire_cost for t in range(num_months))
        layoff_costs_multi = gp.quicksum(L_multi[t] * layoff_cost for t in range(num_months))
        inventory_costs_multi = gp.quicksum(I_multi[t] * inventory_cost for t in range(num_months))
        backorder_costs_multi = gp.quicksum(B_multi[t] * backorder_cost for t in range(num_months))

        total_costs_multi = (material_costs_multi + outsource_costs_multi + labor_regular_costs_multi +
                            labor_overtime_costs_multi + hire_costs_multi + layoff_costs_multi +
                            inventory_costs_multi + backorder_costs_multi)

        profit_multi = revenue_multi - total_costs_multi
        m_multi.setObjective(profit_multi, GRB.MAXIMIZE)

        m_multi.update()
        m_multi.optimize()

        if m_multi.status == GRB.OPTIMAL:
            combo_profit = profit_multi.getValue()
            if best_profit_for_num is None or combo_profit > best_profit_for_num:
                best_profit_for_num = combo_profit
                best_combo_for_num = combo

    if best_combo_for_num is not None:
        multi_promotion_results[num_promos] = {
            'combo': best_combo_for_num,
            'profit': best_profit_for_num,
            'months': [months[m] for m in best_combo_for_num]
        }
        print(f"  最优组合: {[months[m] for m in best_combo_for_num]}")
        print(f"  最优利润: €{best_profit_for_num:,.2f}")
    print()

# 总结多促销分析
print("="*80)
print("问题9：多个促销活动的最优方案总结")
print("="*80)
print()

if multi_promotion_results:
    print(f"{'促销数量':<15} {'促销月份':<40} {'年度利润(€)':<20}")
    print("-" * 75)

    for num_promos in sorted(multi_promotion_results.keys()):
        result = multi_promotion_results[num_promos]
        months_str = ", ".join(result['months'])
        print(f"{num_promos:<15} {months_str:<40} {result['profit']:>18,.2f}")

    print()

    # 找到全局最优方案
    best_overall = max(multi_promotion_results.items(), key=lambda x: x[1]['profit'])
    num_best, best_result_overall = best_overall

    print(f"全局最优方案:")
    print(f"  促销数量: {num_best}")
    print(f"  促销月份: {', '.join(best_result_overall['months'])}")
    print(f"  年度利润: €{best_result_overall['profit']:,.2f}")
    print()

    # 与基础模型和单促销对比
    if m.status == GRB.OPTIMAL:
        base_profit = profit.getValue()
        single_promo_profit = best_result['profit']
        multi_promo_profit = best_result_overall['profit']

        print(f"与基础模型对比:")
        print(f"  基础模型利润:    €{base_profit:,.2f}")
        print(f"  单促销最优利润:  €{single_promo_profit:,.2f} ({(single_promo_profit - base_profit)/base_profit*100:+.2f}%)")
        print(f"  多促销最优利润:  €{multi_promo_profit:,.2f} ({(multi_promo_profit - base_profit)/base_profit*100:+.2f}%)")

        if multi_promo_profit > single_promo_profit:
            improvement = multi_promo_profit - single_promo_profit
            print()
            print(f"多促销相比单促销的改进: €{improvement:,.2f} ({improvement/single_promo_profit*100:+.2f}%)")
        else:
            print()
            print(f"单促销已是最优策略，多促销不会带来额外收益")

print("\n" + "="*80)
