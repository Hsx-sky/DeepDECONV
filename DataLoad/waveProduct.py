import numpy as np
import matplotlib.pyplot as plt


def generate_positive_integers(m, n):
    """
    生成 n 个大于 0 的随机整数，使它们的和等于 m。

    Parameters:
    - m: 随机数的总和
    - n: 随机数的数量

    Returns:
    - integers: 生成的随机整数
    """
    m = m-2

    # 生成随机分隔点
    points = np.sort(np.random.permutation(m+1)[:n - 1])

    # 计算相邻点之间的差值，并调整结果以确保每个值大于 0
    integers = np.diff(np.concatenate(([-1], points, [m+1])))

    return integers


def generate_trapezoidal_wave(length, rise_len, flat_len, fall_len, rise_num, cat_len, cat_int):
    """
    生成梯形波形，包含上升段、平稳段和下降段。

    Parameters:
    - length: 波包的长度
    - rise_len: 上升段的长度
    - flat_len: 平稳段的长度
    - fall_len: 下降段的长度

    Returns:
    - wave: 生成的梯形波形
    """
    # 确保长度匹配
    assert rise_len + flat_len + fall_len == length, "Length mismatch"

    # 生成上升段
    rise_cat = generate_positive_integers(rise_len, (rise_num*2)-1)
    rise_segment = []
    start = 0
    index = 0
    for i in rise_cat:
        index += 1
        if index % 2 == 1:
            rise = np.linspace(start+1, start+i, i)
        else:
            rise = np.linspace(start, start, i)
        start = rise[-1]
        rise_segment = np.append(rise_segment, rise)
    # 生成平稳段
    flat_segment = rise_segment[-1] * np.ones(flat_len)
    # 生成下降段
    fall_segment = np.linspace(rise_segment[-1], 0, fall_len)

    # 合并段
    wave = np.concatenate([[0], rise_segment, flat_segment, fall_segment])
    wave[cat_len:] -= cat_int
    return wave


def add_sine_oscillation(wave, frequency, amplitude):
    """
    在波形上添加正弦震荡。

    Parameters:
    - wave: 基础波形
    - frequency: 正弦波的频率
    - amplitude: 正弦波的振幅

    Returns:
    - wave_with_oscillation: 添加了正弦震荡的波形
    """
    x = np.linspace(0, 2 * np.pi, len(wave))
    oscillation = amplitude * np.sin(frequency * x)
    wave_with_oscillation = wave + oscillation
    return wave_with_oscillation


# 波包长度
'''length = 100
rise_len = np.random.randint(length // 4, length // 2)
flat_len = np.random.randint(length // 4, length // 3)
fall_len = length - rise_len - flat_len'''


def wave_with_oscillation_product(rise_len, flat_len, fall_len, rise_num, cat_len, cat_int):
    # 生成基础梯形波形
    length = rise_len + flat_len + fall_len
    trapezoidal_wave = generate_trapezoidal_wave(length, rise_len, flat_len, fall_len, rise_num, cat_len, cat_int)
    maxA = max(trapezoidal_wave)
    # 随机生成正弦波的频率和振幅
    frequency = np.random.uniform(0.01, 0.2) * 2 * np.pi
    amplitude = np.random.uniform(0.1*maxA, 0.3*maxA)

    # 添加正弦震荡
    wave_with_oscillation = add_sine_oscillation(trapezoidal_wave, frequency, amplitude)
    return wave_with_oscillation, trapezoidal_wave

'''wave_with_oscillation, trapezoidal_wave = wave_with_oscillation_product(10, 10, 10, 2)
# 可视化波形
plt.plot(wave_with_oscillation, label='Wave with Oscillation')
plt.plot(trapezoidal_wave, label='Base Trapezoidal Wave', linestyle='--')
plt.title('Trapezoidal Wave with Random Sine Oscillation')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show(block=True)'''
