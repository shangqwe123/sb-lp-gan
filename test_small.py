import torch
from backbone.ncsnpp.ncsnpp import NCSNpp, NCSNpp12M, NCSNpp6M, NCSNpp2M, NCSNpp150K, NCSNpp40K
from thop import profile

# 定义模型列表和名称
model_classes = [NCSNpp, NCSNpp12M, NCSNpp6M, NCSNpp2M, NCSNpp150K, NCSNpp40K]
model_names = ["NCSNpp", "NCSNpp12M", "NCSNpp6M", "NCSNpp2M", "NCSNpp150K", "NCSNpp40K"]

# 构造复数输入
x_real = torch.randn(1, 4, 256, 256)
x_imag = torch.randn(1, 4, 256, 256)
x = torch.complex(x_real, x_imag)  # (batch, channels, freq, time)，复数
time_cond = torch.ones(1)  # 假设需要time_cond


results = []
for model_class, name in zip(model_classes, model_names):
    model = model_class(input_channels=4, spatial_channels=1, image_size=256)
    model.eval()

    # 进行单次前向推理，并计算rtf（实时系数），即推理总时长 / 原始信号时长
    import time
    x_test = x
    t_test = time_cond

    with torch.no_grad():
        elapsed_times = []
        num_runs = 10  # 运行次数
        for _ in range(num_runs):
            start_time = time.time()
            out = model(x_test, t_test)
            end_time = time.time()
            elapsed_times.append(end_time - start_time)
    mean_elapsed = sum(elapsed_times) / num_runs  # 平均推理时长（秒）

    rtf = mean_elapsed
    print(f"{name} 平均RTF (real time factor): {rtf:.5f}，{num_runs}次均值")

    try:
        flops, params = profile(model, inputs=(x, time_cond), verbose=False)
        result = f"{name} 单次前向 FLOPs: {flops / 1e9:.2f} GFLOPs\n{name} MACs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M\n"
    except Exception as e:
        result = f"{name} 计算失败: {e}\n"
    print(result)
    results.append(result)

with open("result.txt", "w", encoding="utf-8") as f:
    f.writelines(results)
