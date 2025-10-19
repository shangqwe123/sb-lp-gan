from MISB.sdes import DiffusionBridgeSDE
from MISB.plot import plot_diffusion_coefficients

# 创建一个VP调度的SDE实例
sde_vp = DiffusionBridgeSDE(schedule_type='VP')
# 创建一个VE调度的SDE实例
sde_ve = DiffusionBridgeSDE(schedule_type='VE')
# 创建一个gmax调度的SDE实例
sde_gmax = DiffusionBridgeSDE(schedule_type='gmax')

# 绘制三种调度类型的系数变化
plot_diffusion_coefficients(sde_vp)
plot_diffusion_coefficients(sde_ve)
plot_diffusion_coefficients(sde_gmax)