import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Исходные данные
m_o = 1000  # масса объекта без топлива (кг)
m_f0 = 500  # начальная масса топлива (кг)
dot_m_f = 5  # расход топлива (кг/с)
F_engine = 2000  # сила двигателя (Н)
v_0 = 0  # начальная скорость (м/с)

# Параметры замедления
C_d = 0.47  # коэффициент сопротивления воздуха
rho = 1.225  # плотность воздуха (кг/м^3)
A = 0.5  # площадь поперечного сечения (м^2)
mu = 0.135  # коэффициент трения
g = 9.81  # ускорение свободного падения (м/с^2)

# Время расчета для ускорения
t_max = m_f0 / dot_m_f  # время полного сгорания топлива
dt = 1  # шаг интегрирования (с)
t_accel = np.arange(0, t_max + dt, dt)  # временной массив для ускорения

# Инициализация переменных для ускорения
v_accel = np.zeros_like(t_accel)  # скорость при ускорении
v_accel[0] = v_0 + 0.00001  # начальная скорость при ускорении
m_accel = m_o + m_f0 - dot_m_f * t_accel  # масса объекта при ускорении
a_accel = np.zeros_like(t_accel)  # ускорение при ускорении
s_accel = np.zeros_like(t_accel)  # расстояние при ускорении

# Расчёт данных для ускорения
for i in range(1, len(t_accel)):
    if m_accel[i] > m_o:
        # Сила сопротивления от воздуха
        F_air_resistance = 0.5 * C_d * rho * A * v_accel[i-1]**2
        # Сила трения
        F_friction = mu * m_accel[i] * g
        # Общая сила сопротивления
        F_resistance = F_air_resistance + F_friction
        # Ускорение
        a_accel[i] = (F_engine - F_resistance) / m_accel[i]
    else:
        # После окончания топлива только сопротивление воздуха и трение
        F_air_resistance = 0.5 * C_d * rho * A * v_accel[i-1]**2
        F_friction = mu * m_accel[i] * g
        F_resistance = F_air_resistance + F_friction
        a_accel[i] = -F_resistance / m_accel[i]  # замедление после окончания топлива

    v_accel[i] = v_accel[i-1] + a_accel[i] * dt
    s_accel[i] = s_accel[i-1] + v_accel[i] * dt

    if v_accel[i] < 0:
        v_accel[i] = 0
        break

# Начальная скорость для замедления
v_decel_0 = v_accel[-1]

# Время моделирования для замедления
t_max_decel = 100  # максимальное время моделирования (с)
t_decel = np.arange(0, t_max_decel, dt)  # временной массив для замедления

# Инициализация переменных для замедления
v_decel = np.zeros(len(t_decel))  # скорость при замедлении
v_decel[0] = v_decel_0  # начальная скорость для замедления
m_decel = m_o + m_f0 - dot_m_f * t_max  # масса при замедлении (масса объекта + топливо)
s_decel = np.zeros(len(t_decel))  # расстояние при замедлении

# Численное решение для замедления
for i in range(1, len(t_decel)):
    # Сила сопротивления от воздуха
    F_air_resistance = 0.5 * C_d * rho * A * v_decel[i-1]**2
    # Сила трения
    F_friction = mu * m_decel * g
    # Общая сила сопротивления
    F_resistance = F_air_resistance + F_friction
    # Ускорение (замедление)
    a_decel = -F_resistance / m_decel
    v_decel[i] = v_decel[i-1] + a_decel * dt  # обновление скорости
    s_decel[i] = s_decel[i-1] + v_decel[i-1] * dt  # обновление расстояния

    if v_decel[i] < 0:
        v_decel[i] = 0  # скорость не может быть отрицательной

# Объединение данных ускорения и замедления
t = np.concatenate((t_accel, t_decel + t_accel[-1]))
v = np.concatenate((v_accel, v_decel))
s = np.concatenate((s_accel, s_decel + s_accel[-1]))

# Построение графика
fig, ax = plt.subplots(figsize=(10, 6))

# Настройка графика анимации
line, = ax.plot([], [], 'b-', lw=2, label="Скорость (м/с)")  # Пустой график
ax.set_xlim(0, max(t))
ax.set_ylim(0, max(v) * 1.1)
ax.set_xlabel("Время (с)")
ax.set_ylabel("Скорость (м/с)")
ax.grid(True)
ax.legend()

# Текст для отображения текущего расстояния
distance_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

# Данные для анимации
x_data, y_data = [], []

# Функция обновления для анимации
def update(frame):
    if v[frame] <= 0:
        ani.event_source.stop()  # остановка анимации после завершения
        return line, distance_text

    x_data.append(t[frame])
    y_data.append(v[frame])
    line.set_data(x_data, y_data)
    distance_text.set_text(f'Расстояние: {s[frame]:.2f} м')
    return line, distance_text

# Настройка анимации
ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

# Отображение графика
plt.show()
