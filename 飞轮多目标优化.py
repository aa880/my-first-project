# -*- coding: utf-8 -*-
# 飞轮多目标优化设计系统 v4.0 - 舍弃斜面角度参数，修正临界转速计算

# 首先导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.config import Config
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import logging
import os

# ==================== 配置日志记录 ====================
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("flywheel_optimizer")

# ==================== 可视化参数设置 ====================
# 设置中文字体（解决绘图时中文乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 修复坐标轴负号显示异常问题
plt.rcParams['axes.unicode_minus'] = False
# 设置全局网格样式
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['grid.linewidth'] = 0.5
# 禁用未编译代码的警告提示
Config.warnings['not_compiled'] = False

# 创建结果目录
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/evolution'):
    os.makedirs('results/evolution')


# ==================== 飞轮优化问题定义 ====================
class FlywheelProblem(Problem):
    """定义飞轮多目标优化问题（继承自pymoo的Problem基类）"""

    def __init__(self):
        # 调用父类初始化方法
        super().__init__(
            n_var=3,  # 变量维度：[K衰减系数, 外径(m), 初始厚度(m)]
            n_obj=3,  # 目标维度：储能密度↑ 最大储能量↑ 应力比↓
            n_constr=4,  # 约束维度：临界转速/最小厚度/储能量/厚度差异
            xl=np.array([0.20, 0.15, 0.08]),  # 变量下限（对应K,外径,厚度）
            xu=np.array([0.45, 0.18, 0.13])  # 变量上限（同上顺序）
        )
        self.r0 = 0.03  # 内径=30mm（对应图纸规格）
        self.rho = 7800  # 材料密度=7.8g/cm³（典型钢材密度）
        self.sigma_allow = 1.0e9  # 许用应力=1.0GPa（高强度合金钢）
        self.omega = 2094.4  # 转速20000rpm转换后的弧度值(20000 * 2π/60) = 2094.4
        self.E = 209e9  # 弹性模量=209GPa（钢材典型值）
        self.mu = 0.3  # 泊松比（金属材料常见值）
        self.n_segments = 5  # 中间环段数量（对应阶梯结构5段）
        self.min_thickness_diff = 0.005  # 相邻环段最小厚度差异为5mm

    def _evaluate(self, X, out, *args, **kwargs):
        """多目标评估函数（核心计算逻辑）
        参数：
            X : 决策变量矩阵 [n_samples, n_var]
            out : 输出字典包含目标F和约束G
        """
        n_samples = X.shape[0]

        # 向量化初始化输出数组
        F = np.zeros((n_samples, self.n_obj))
        G = np.zeros((n_samples, self.n_constr))

        # 遍历种群中的每个个体（设计解）
        for idx, x in enumerate(X):
            K, re, t0 = x  # 解包设计变量（移除theta）

            # @@ 设计有效性检查（防止外径过小）
            if re < self.r0 + 0.05:  # 外径必须≥内径+5cm（保证结构合理性）
                F[idx] = [0, 1e6, 1e6]  # 无效解目标值设置为极大劣化值
                G[idx] = [1e6, 1e6, 1e6, 1e6]  # 约束违反标记
                continue  # 跳过后续计算

            # ========== 几何建模阶段 ==========
            # @@ 半径生成（使用指数衰减模型）@@
            radii = [self.r0]  # 存储各环段内径的列表，初始包含内径
            for i in range(1, self.n_segments + 1):
                denominator = 1 - np.exp(-K * self.n_segments)  # 分母归一化项
                term = (1 - np.exp(-K * i)) / denominator  # 比例系数计算
                radius = self.r0 + (re - self.r0) * term  # 当前半径计算
                radii.append(radius)

            # @@ 计算各段中点半径 @@
            r_mid_points = []
            for i in range(len(radii) - 1):
                r_mid = (radii[i] + radii[i + 1]) / 2
                r_mid_points.append(r_mid)

            # @@ 计算基础曲线在中点处的理想厚度值 @@
            t_ideal = [t0 * (r / self.r0) ** -K for r in r_mid_points]

            # @@ 根据基础厚度曲线生成阶梯状厚度分布 @@
            t = []
            t.append(t_ideal[0])  # 第一段使用理想厚度

            # 计算后续环段厚度，确保厚度单调递减且差异显著
            for i in range(1, len(t_ideal)):
                # 确保每段厚度比前一段至少小min_thickness_diff
                proposed_t = min(t_ideal[i], t[i - 1] - self.min_thickness_diff)
                t.append(max(proposed_t, 0.002))  # 保证最小厚度约束

            # 计算厚度差异是否满足要求
            min_diff = float('inf')
            for i in range(1, len(t)):
                diff = t[i - 1] - t[i]  # 厚度差值（应为正）
                min_diff = min(min_diff, diff)

            # ========== 物理量计算阶段 ==========
            mass, J = 0.0, 0.0  # 初始化总质量和转动惯量
            for i in range(len(t)):
                r_in, r_out = radii[i], radii[i + 1]  # 当前环段的内外径
                # @@ 环段质量计算（圆环体积公式） @@
                volume = np.pi * (r_out ** 2 - r_in ** 2) * t[i]  # 环段体积
                mass += self.rho * volume  # 累加总质量
                # @@ 转动惯量计算（圆环公式） @@
                J += 0.5 * self.rho * np.pi * t[i] * (r_out ** 4 - r_in ** 4)

            # ========== 性能指标计算 ==========
            # 储能量计算（动能公式，转换为Wh单位）
            energy_wh = (0.5 * J * self.omega ** 2) / 3600

            # 应力计算
            # 最大周向应力（基于变厚度旋转盘改进公式）
            max_stress = (3 + self.mu) / 8 * self.rho * self.omega ** 2 * re ** 2 * (1 - (self.r0 / re) ** 2)
            stress_ratio = max_stress / self.sigma_allow  # 应力安全系数

            # ========== 约束条件计算 ==========
            try:
                # 临界转速计算（防止共振）- 修正版
                avg_thickness = sum(t) / len(t)  # 平均厚度(m)
                effective_radius = (re + self.r0) / 2  # 有效半径(m)

                # 使用校正的临界转速公式
                # 考虑变厚度圆盘的特性
                correction_factor = 0.65  # 工程校正系数
                # 公式单位:
                # E(Pa), rho(kg/m³), thickness(m), radius(m)
                # 输出单位: rad/s
                n_cr = correction_factor * np.sqrt(self.E / self.rho) * avg_thickness / (effective_radius ** 2)

                # 转换为rpm
                n_cr = n_cr / (2 * np.pi) * 60
            except:
                n_cr = 0  # 处理数学异常（如除零）

            # 约束条件：
            # 1. 临界转速≥23000rpm → 23000 - n_cr ≤0
            # 2. 最小厚度≥2mm → 0.002 - min(t) ≤0
            # 3. 储能量≥700Wh → 700 - energy_wh ≤0
            # 4. 厚度差异≥5mm → self.min_thickness_diff - min_diff ≤0
            G[idx] = [
                23000 - n_cr,
                0.002 - min(t) if t else 1e6,
                700 - energy_wh,
                self.min_thickness_diff - min_diff if min_diff != float('inf') else 0  # 厚度差异约束
            ]

            # ========== 目标函数赋值 ==========
            F[idx] = [
                -energy_wh / mass if mass > 0 else 1e6,  # 目标1：储能密度（Wh/kg）最大化（取负转为最小化）
                -energy_wh,  # 目标2：储能量（Wh）最大化（取负转为最小化）
                stress_ratio  # 目标3：应力安全系数最小化
            ]

        # 输出计算结果
        out["F"] = F  # 目标函数矩阵
        out["G"] = G  # 约束条件矩阵


# ==================== 优化过程监控器 ====================
class 优化监控器:
    """实时显示优化过程的监控界面（适配NSGA-II算法）"""

    def __init__(self, problem):
        # 表格列定义（与可视化布局严格对齐）
        self.表头 = ["进化代数", "评估次数", "非劣解数", "最小约束违反", "平均约束违反", "收敛阈值", "性能指标"]
        self.列宽 = [6, 8, 8, 12, 12, 8, 12]  # 各列字符宽度
        self.problem = problem
        self.visualization_count = 0  # 可视化计数器
        self.gen_history = []  # 代数历史记录
        self.pareto_history = []  # Pareto前沿历史记录
        self.打印表头()  # 初始化时打印表头

    def 打印表头(self):
        """绘制控制台表格的顶部结构"""
        # 顶部边框线
        print("╭" + "┬".join(["─" * w for w in self.列宽]) + "╮")
        # 表头内容居中显示
        print("│" + "│".join(f"{h:^{w}}" for h, w in zip(self.表头, self.列宽)) + "│")
        # 表头与数据分隔线
        print("├" + "┼".join(["─" * w for w in self.列宽]) + "┤")

    def 更新数据(self, 算法):
        # 提取关键指标数据
        数据 = [
            算法.n_gen,  # 当前进化代数
            算法.evaluator.n_eval,  # 已评估解的总数
            len(算法.opt),  # 非支配解的数量
            np.min(算法.pop.get("CV")) if len(算法.pop) > 0 else 1.0,  # 种群中最小约束违反值
            np.mean(算法.pop.get("CV")) if len(算法.pop) > 0 else 1.0,  # 平均约束违反程度
            1e-6,  # 收敛判断阈值（固定值）
            f"非劣解: {len(算法.opt)}"  # 性能指标
        ]

        # 记录日志
        logger.info(f"代数: {数据[0]}, 评估次数: {数据[1]}, 非劣解数: {数据[2]}, 最小约束违反: {数据[3]:.4f}")

        # 格式化表格行数据
        格式行 = "│" + "│".join([
            f"{数据[0]:^{self.列宽[0]}.0f}",  # 代数（整数格式）
            f"{数据[1]:^{self.列宽[1]}}",  # 评估次数（原样输出）
            f"{数据[2]:^{self.列宽[2]}.0f}",  # 非劣解数（整数）
            f"{数据[3]:^{self.列宽[3]}.4f}",  # 最小约束违反（4位小数）
            f"{数据[4]:^{self.列宽[4]}.4f}",  # 平均约束违反
            f"{数据[5]:^{self.列宽[5]}.1e}",  # 收敛阈值（科学计数法）
            f"{数据[6]:^{self.列宽[6]}}"  # 性能指标（字符串）
        ]) + "│"
        print(格式行)

        # 每50代进行一次Pareto前沿可视化
        if 算法.n_gen % 50 == 0:
            self.可视化进化过程(算法)

    def 可视化进化过程(self, algorithm):
        """可视化NSGA-II的进化过程（每50代）"""
        F = algorithm.pop.get("F")  # 获取当前种群的目标值矩阵

        # 从当前种群中选择非支配解
        nds = NonDominatedSorting()
        front_idx = nds.do(F, only_non_dominated_front=True)
        pareto_front = F[front_idx]

        # 记录历史数据
        self.gen_history.append(algorithm.n_gen)
        self.pareto_history.append(pareto_front)

        # 目标0和目标1的关系（储能密度 vs 储能量）
        fig = plt.figure(figsize=(12, 10))
        plt.suptitle(f"NSGA-II进化过程 (第{algorithm.n_gen}代)", fontsize=16, y=0.98)

        # 稀疏采样设置
        sample_rate = 5  # 每隔5个点取样一次
        sample_indices = np.arange(0, len(F), sample_rate)
        sampled_F = F[sample_indices]

        # 储能密度 vs 储能量
        plt.subplot(221)
        plt.scatter([-f[0] for f in sampled_F], [-f[1] for f in sampled_F], s=10, alpha=0.4, c='gray',
                    label=f'所有个体 (每{sample_rate}个取1个)')
        plt.scatter([-f[0] for f in pareto_front], [-f[1] for f in pareto_front], s=20, c='red', edgecolor='k',
                    alpha=0.7, label='Pareto前沿')
        plt.xlabel('储能密度 (Wh/kg)')
        plt.ylabel('储能量 (Wh)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(MaxNLocator(5))
        plt.gca().yaxis.set_major_locator(MaxNLocator(5))

        # 储能密度 vs 应力比
        plt.subplot(222)
        plt.scatter([-f[0] for f in sampled_F], [f[2] for f in sampled_F], s=10, alpha=0.4, c='gray',
                    label=f'所有个体 (每{sample_rate}个取1个)')
        plt.scatter([-f[0] for f in pareto_front], [f[2] for f in pareto_front], s=20, c='red', edgecolor='k',
                    alpha=0.7, label='Pareto前沿')
        plt.xlabel('储能密度 (Wh/kg)')
        plt.ylabel('应力比')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(MaxNLocator(5))
        plt.gca().yaxis.set_major_locator(MaxNLocator(5))

        # 储能量 vs 应力比
        plt.subplot(223)
        plt.scatter([-f[1] for f in sampled_F], [f[2] for f in sampled_F], s=10, alpha=0.4, c='gray',
                    label=f'所有个体 (每{sample_rate}个取1个)')
        plt.scatter([-f[1] for f in pareto_front], [f[2] for f in pareto_front], s=20, c='red', edgecolor='k',
                    alpha=0.7, label='Pareto前沿')
        plt.xlabel('储能量 (Wh)')
        plt.ylabel('应力比')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(MaxNLocator(5))
        plt.gca().yaxis.set_major_locator(MaxNLocator(5))

        # 目标函数分布
        plt.subplot(224)
        objectives = ['储能密度', '储能量', '应力比']
        values = [-np.mean(F[:, 0]), -np.mean(F[:, 1]), np.mean(F[:, 2])]
        std_devs = [np.std(-F[:, 0]), np.std(-F[:, 1]), np.std(F[:, 2])]

        plt.bar(objectives, values, yerr=std_devs, alpha=0.7, capsize=5)
        plt.title('目标函数平均值')
        plt.ylabel('目标值')
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/evolution/nsga2_gen_{algorithm.n_gen}.png', dpi=100)
        plt.close()

        # 保存本次可视化结果
        self.visualization_count += 1
        logger.info(f"保存第{algorithm.n_gen}代优化可视化图为: results/evolution/nsga2_gen_{algorithm.n_gen}.png")


# ==================== 可视化厚度分布函数 ====================
def 可视化厚度分布(radii, t, K, re, t0, filename):
    """可视化厚度分布与半径的关系"""
    plt.figure(figsize=(10, 6))

    # 厚度分布散点图
    r_points = [(radii[i] + radii[i + 1]) / 2 for i in range(len(t))]  # 每段中点
    plt.scatter(r_points, [ti * 1000 for ti in t], s=100, color='blue', zorder=3, label='计算厚度点')

    # 连线
    plt.plot(r_points, [ti * 1000 for ti in t], 'b--', alpha=0.7, zorder=2)

    # 理论曲线 - 基础厚度曲线
    r_continuous = np.linspace(radii[0], radii[-1], 100)
    t_base = [t0 * 1000 * (r / radii[0]) ** -K for r in r_continuous]
    plt.plot(r_continuous, t_base, 'r-', alpha=0.5, label='基础厚度曲线')

    # 绘制垂直线标记每段边界
    for r in radii:
        plt.axvline(x=r, color='gray', linestyle=':', alpha=0.5)

    # 添加环段标签和厚度标注
    for i in range(len(t)):
        plt.text((radii[i] + radii[i + 1]) / 2, t[i] * 1000 + 5, f'段{i + 1}',
                 ha='center', va='bottom', fontsize=10)
        # 添加厚度值标注
        plt.text((radii[i] + radii[i + 1]) / 2, t[i] * 1000 - 5, f'{t[i] * 1000:.1f}mm',
                 ha='center', va='top', fontsize=9, color='darkblue')

    # 添加厚度差异标注
    for i in range(1, len(t)):
        diff = (t[i - 1] - t[i]) * 1000  # 厚度差 (mm)
        mid_r = (r_points[i - 1] + r_points[i]) / 2
        mid_t = (t[i - 1] * 1000 + t[i] * 1000) / 2
        plt.annotate(f'Δ{diff:.1f}mm', xy=(mid_r, mid_t), xytext=(mid_r, mid_t + 10),
                     arrowprops=dict(arrowstyle='->'), ha='center', fontsize=8)

    plt.title(f'飞轮厚度分布曲线 (K={K:.3f}, 最小厚度差={5.0}mm)')
    plt.xlabel('半径 (m)')
    plt.ylabel('厚度 (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ==================== 主程序执行流程 ====================
if __name__ == "__main__":
    logger.info("开始飞轮多目标优化设计 v4.0 - 移除斜面角度参数")

    # 初始化问题
    problem = FlywheelProblem()
    # 初始化监控器
    监控器 = 优化监控器(problem)

    # 配置优化算法参数
    algorithm = NSGA2(
        pop_size=200,  # 种群规模
        crossover=SBX(prob=0.85, eta=20),  # 交叉概率85%，分布指数20（控制交叉强度）
        mutation=PM(eta=25),  # 变异算子，分布指数25（控制变异幅度）
        eliminate_duplicates=True,  # 开启解去重功能
        callback=监控器.更新数据  # 注册监控回调函数
    )

    # 执行优化过程（运行500代）
    logger.info("开始优化计算...")
    result = minimize(
        problem,
        algorithm,
        ('n_gen', 500),  # 优化终止条件：500代
        seed=42,  # 随机种子（保证结果可复现）
        verbose=False  # 关闭算法原生输出
    )
    logger.info("优化计算完成")

    # ========== 优化结果后处理 ==========
    if result.G is not None:
        feasible = np.all(result.G <= 0, axis=1)
        可行解 = result.X[feasible]
        F_vals = result.F[feasible]

        if len(可行解) == 0:
            logger.warning("未找到满足约束的可行解！")
            # 选择违反约束最小的解
            min_cv_idx = np.argmin(np.sum(np.maximum(0, result.G), axis=1))
            best_solution = result.X[min_cv_idx]
            logger.info(f"选择约束违反最小的解: CV = {np.sum(np.maximum(0, result.G[min_cv_idx])):.4f}")
            K_opt, re_opt, t0_opt = best_solution
        else:
            logger.info(f"找到{len(可行解)}个可行解")
            # 选择储能密度最大的解
            best_idx = np.argmin(F_vals[:, 0])  # 最小化-储能密度
            K_opt, re_opt, t0_opt = 可行解[best_idx]
            logger.info(f"最优解: 储能密度={-F_vals[best_idx, 0]:.2f} Wh/kg, 储能量={-F_vals[best_idx, 1]:.1f} Wh")

        r0 = problem.r0
        n_segments = problem.n_segments

        # ====== 统一半径生成方式 ======
        radii = [r0]  # 从内径开始
        for i in range(1, n_segments + 1):
            denominator = 1 - np.exp(-K_opt * n_segments)
            term = (1 - np.exp(-K_opt * i)) / denominator
            radius = r0 + (re_opt - r0) * term
            radii.append(radius)

        # ====== 计算各段中点半径 ======
        r_mid_points = []
        for i in range(len(radii) - 1):
            r_mid = (radii[i] + radii[i + 1]) / 2
            r_mid_points.append(r_mid)

        # ====== 计算基础曲线在中点处的理想厚度值 ======
        t_ideal = [t0_opt * (r / r0) ** -K_opt for r in r_mid_points]

        # ====== 根据基础厚度曲线生成阶梯状厚度分布 ======
        t = []
        t.append(t_ideal[0])  # 第一段使用理想厚度

        # 计算后续环段厚度，确保厚度单调递减且差异显著
        for i in range(1, len(t_ideal)):
            # 确保每段厚度比前一段至少小min_thickness_diff
            proposed_t = min(t_ideal[i], t[i - 1] - problem.min_thickness_diff)
            t.append(max(proposed_t, 0.002))  # 保证最小厚度约束

        # ====== 数据校验 ======
        assert len(radii) == n_segments + 1, f"半径数量异常: {len(radii)}"
        assert len(t) == n_segments, f"厚度数量异常: {len(t)}"

        # ====== 可视化厚度分布 ======
        可视化厚度分布(radii, t, K_opt, re_opt, t0_opt, 'results/thickness_distribution.png')
        logger.info("保存厚度分布图为: results/thickness_distribution.png")

        # ====== 计算性能指标 ======
        # 计算质量和体积
        total_mass, total_volume = 0.0, 0.0
        for i in range(n_segments):
            r_in, r_out = radii[i], radii[i + 1]
            volume = np.pi * (r_out ** 2 - r_in ** 2) * t[i]
            total_volume += volume
            total_mass += volume * problem.rho

        # 计算转动惯量和储能量
        J = 0.0
        for i in range(n_segments):
            r_in, r_out = radii[i], radii[i + 1]
            J += 0.5 * problem.rho * np.pi * t[i] * (r_out ** 4 - r_in ** 4)
        energy_wh = (0.5 * J * problem.omega ** 2) / 3600
        energy_density = energy_wh / total_mass

        # 计算临界转速 - 使用修正后的公式
        try:
            avg_thickness = sum(t) / len(t)  # 平均厚度(m)
            effective_radius = (re_opt + r0) / 2  # 有效半径(m)

            # 使用校正的临界转速公式
            correction_factor = 0.65  # 工程校正系数
            n_cr = correction_factor * np.sqrt(problem.E / problem.rho) * avg_thickness / (effective_radius ** 2)

            # 转换为rpm
            n_cr = n_cr / (2 * np.pi) * 60
        except:
            n_cr = 0

        # 计算最大应力和应力比
        max_stress = (3 + problem.mu) / 8 * problem.rho * problem.omega ** 2 * re_opt ** 2 * (1 - (r0 / re_opt) ** 2)
        stress_ratio = max_stress / problem.sigma_allow

        # ====== 终端输出 ======
        logger.info(f"\n最优解参数：K={K_opt:.3f} 外径={re_opt:.3f}m")
        logger.info(f"初始厚度={t0_opt * 1000:.2f}mm")
        logger.info(f"质量：{total_mass:.2f}kg 储能量：{energy_wh:.1f}Wh")
        logger.info(f"储能密度：{energy_density:.1f} Wh/kg 应力比：{stress_ratio:.3f}")
        logger.info(f"临界转速：{n_cr:.0f} rpm")

        # ====== 生成环段参数表 ======
        # 生成环段数据（带ΔR计算）
        step_table_data = [["环段", "内径(mm)", "外径(mm)", "ΔR(mm)", "厚度(mm)"]]
        for i in range(len(radii) - 1):
            delta_r = (radii[i + 1] - radii[i]) * 1000
            step_table_data.append([
                f"第{i + 1}段",
                f"{radii[i] * 1000:.1f}",
                f"{radii[i + 1] * 1000:.1f}",
                f"{delta_r:.1f}",
                f"{t[i] * 1000:.2f}"
            ])

        # ====== 可视化参数表格 ======
        fig = plt.figure(figsize=(12, 10), facecolor='#F0F0F0')  # 设置浅灰背景
        gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.4)  # 上下布局比例

        # ====== 主参数表 ======
        ax1 = fig.add_subplot(gs[0])
        ax1.axis('off')

        # 双列布局数据（左侧结构参数，右侧性能参数）
        main_table_data = [
            ["参数", "数值", "单位", "参数", "数值", "单位"],
            ["衰减系数 K", f"{K_opt:.3f}", "-", "储能量", f"{energy_wh:.1f}", "Wh"],
            ["外径", f"{re_opt * 1000:.1f}", "mm", "储能密度", f"{energy_density:.1f}", "Wh/kg"],
            ["内径", f"{r0 * 1000:.1f}", "mm", "应力比", f"{stress_ratio:.3f}", "-"],
            ["初始厚度", f"{t0_opt * 1000:.2f}", "mm", "临界转速", f"{n_cr:.0f}", "rpm"],
            ["最小厚度差", f"{problem.min_thickness_diff * 1000:.1f}", "mm", "最小厚度", f"{min(t) * 1000:.2f}", "mm"],
            ["总质量", f"{total_mass:.2f}", "kg", "最大厚度", f"{max(t) * 1000:.2f}", "mm"],
            ["总体积", f"{total_volume:.6f}", "m³", "厚度比", f"{min(t) / max(t):.3f}", "-"]
        ]

        # 创建主表格（6列布局）
        main_table = ax1.table(
            cellText=main_table_data,
            colWidths=[0.18, 0.15, 0.1, 0.18, 0.15, 0.1],  # 列宽匹配图片比例
            cellLoc='center',
            loc='center',
            edges='horizontal',  # 仅显示水平线
            bbox=[0, 0.2, 1, 0.7]  # 表格位置调整
        )

        # 表格样式优化
        for key, cell in main_table.get_celld().items():
            cell.set_edgecolor('#404040')  # 中灰色边框
            if key[1] in [0, 3]:  # 参数名称列
                cell.set_facecolor('#E8E8E8')  # 浅灰色背景
            if key[0] == 0:  # 表头行
                cell.set_facecolor('#D0D0D0')  # 深灰色背景
                cell.set_text_props(weight='bold')

        # ====== 环段参数表 ======
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')

        # 创建阶梯参数表
        step_table = ax2.table(
            cellText=step_table_data,
            colWidths=[0.2, 0.2, 0.2, 0.2, 0.2],  # 五等分列宽
            cellLoc='center',
            loc='center',
            edges='horizontal'
        )

        # 统一表格样式
        for key, cell in step_table.get_celld().items():
            cell.set_edgecolor('#404040')
            if key[0] == 0:  # 表头行
                cell.set_facecolor('#D0D0D0')
                cell.set_text_props(weight='bold')

        fig.suptitle('飞轮优化设计参数表 (v4.0 - 无斜面角度)', fontsize=16, y=0.98)
        plt.savefig('results/final_flywheel_design.png', dpi=120, bbox_inches='tight')
        plt.close()
        logger.info("保存最终设计参数表为: results/final_flywheel_design.png")

        # ====== 创建飞轮剖面图 ======
        plt.figure(figsize=(8, 6))

        # 绘制轮廓
        theta_vals = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(radii)):
            x_outer = radii[i] * np.cos(theta_vals)
            y_outer = radii[i] * np.sin(theta_vals)
            plt.plot(x_outer, y_outer, 'k-', linewidth=1)

        # 绘制厚度
        for i in range(len(t)):
            # 上表面 - 直接使用固定厚度
            x1 = radii[i] * np.cos(np.pi / 4)
            y1 = radii[i] * np.sin(np.pi / 4)
            x2 = radii[i + 1] * np.cos(np.pi / 4)
            y2 = radii[i + 1] * np.sin(np.pi / 4)
            plt.plot([x1, x2], [y1 + t[i], y2 + t[i]], 'b-', linewidth=2)

            # 下表面
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

            # 连接线
            plt.plot([x1, x1], [y1, y1 + t[i]], 'b-', linewidth=1.5, alpha=0.7)
            plt.plot([x2, x2], [y2, y2 + t[i]], 'b-', linewidth=1.5, alpha=0.7)

            # 标记厚度
            plt.text((x1 + x2) / 2, (y1 + y2) / 2 + t[i] / 2, f"{t[i] * 1000:.1f}mm",
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.axis('equal')
        plt.title('飞轮剖面示意图 (阶梯状厚度分布)')
        plt.grid(True, alpha=0.3)
        plt.savefig('results/flywheel_section.png', dpi=150)
        plt.close()
        logger.info("保存飞轮剖面图为: results/flywheel_section.png")

        # ====== 创建3D模型可视化 ======
        # 仅在具有可行解时执行3D可视化
        if len(可行解) > 0:
            # 创建更美观的配色方案
            stress_colors = ["#2b8cbe", "#a8ddb5", "#fed976", "#f03b20"]
            cmap = LinearSegmentedColormap.from_list("stress_map", stress_colors)
            norm = Normalize(vmin=np.min(F_vals[:, 2]), vmax=np.max(F_vals[:, 2]))

            # 创建图形和三维坐标系
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            # 计算三维点云的中心位置
            centers = np.array([
                np.mean(-F_vals[:, 0]),  # 储能密度均值
                np.mean(-F_vals[:, 1]),  # 储能量均值
                np.mean(F_vals[:, 2])  # 应力比均值
            ])

            # 稀疏采样
            sample_rate = 5  # 每5个点取1个
            sample_indices = np.arange(0, len(F_vals), sample_rate)
            sampled_F = F_vals[sample_indices]

            # 绘制稀疏点云
            sc = ax.scatter(
                -sampled_F[:, 0],  # 储能密度
                -sampled_F[:, 1],  # 储能量
                sampled_F[:, 2],  # 应力比
                c=sampled_F[:, 2],  # 按应力比颜色编码
                cmap=cmap,
                s=60,  # 点尺寸
                alpha=0.8,  # 透明度
                edgecolor='w',  # 边框
                linewidth=0.8,  # 边框宽度
                depthshade=True,  # 深度阴影
                label=f'可行解（每{sample_rate}个取1个）'
            )

            # 最优解标记
            ax.scatter(
                -F_vals[best_idx, 0],
                -F_vals[best_idx, 1],
                F_vals[best_idx, 2],
                s=300,  # 尺寸
                c='gold',  # 颜色
                marker='*',  # 标记
                edgecolor='black',  # 边框
                linewidth=1.5,  # 边框宽度
                alpha=1.0,  # 不透明
                label='最优解（储能密度最大）',
                zorder=100  # 层级
            )

            # 设置坐标轴和标签
            ax.set_xlabel('\n储能密度 (Wh/kg)', linespacing=3.2, fontsize=12)
            ax.set_ylabel('\n储能量 (Wh)', linespacing=3.2, fontsize=12)
            ax.set_zlabel('\n应力比', linespacing=3.2, fontsize=12)

            # 添加色条
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=20, pad=0.12)
            cbar.set_label('应力比 (值越低越好)', rotation=270, labelpad=20, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

            # 设置坐标系样式
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.05)
            ax.yaxis.pane.set_alpha(0.05)
            ax.zaxis.pane.set_alpha(0.05)

            # 调整坐标轴刻度
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.zaxis.set_major_locator(MaxNLocator(5))

            # 设置视图角度
            ax.view_init(elev=28, azim=-145)

            # 添加标题
            plt.title('飞轮多目标优化结果 - 三维可视化', fontsize=16, pad=20)

            # 添加图例
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=11,
                      framealpha=0.95, fancybox=True, shadow=True)

            # 添加坐标轴参考平面
            max_range = np.array([
                max(-F_vals[:, 0]) - min(-F_vals[:, 0]),
                max(-F_vals[:, 1]) - min(-F_vals[:, 1]),
                max(F_vals[:, 2]) - min(F_vals[:, 2])
            ]).max() * 0.55

            mid_x = centers[0]
            mid_y = centers[1]
            mid_z = centers[2]

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # 添加参考线
            ax.plot([mid_x - max_range, mid_x + max_range], [mid_y, mid_y], [mid_z, mid_z],
                    'gray', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.plot([mid_x, mid_x], [mid_y - max_range, mid_y + max_range], [mid_z, mid_z],
                    'gray', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.plot([mid_x, mid_x], [mid_y, mid_y], [mid_z - max_range, mid_z + max_range],
                    'gray', linestyle='--', linewidth=0.8, alpha=0.6)

            # 设置字体大小
            for t in ax.xaxis.get_ticklabels(): t.set_fontsize(10)
            for t in ax.yaxis.get_ticklabels(): t.set_fontsize(10)
            for t in ax.zaxis.get_ticklabels(): t.set_fontsize(10)

            # 添加辅助文字
            ax.text(
                mid_x + max_range * 1.1, mid_y - max_range, mid_z,
                "理想区域：高储能密度、高储能量、低应力比",
                fontsize=12, ha='left', va='center', color='darkred'
            )

            plt.savefig('results/flywheel_3d_visualization.png', dpi=160, bbox_inches='tight')
            plt.close()
            logger.info("保存三维可视化图为: results/flywheel_3d_visualization.png")

        # 输出优化完成信息
        logger.info("优化分析完成，所有结果已保存到results目录")
    else:
        logger.error("优化结果异常，未能获取约束信息")
