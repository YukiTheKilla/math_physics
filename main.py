import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Исходные данные
m_o = 1000  # масса объекта без топлива (кг)
m_f0 = 500  # начальная масса топлива (кг)
dot_m_f = 5  # расход топлива (кг/с)
F_engine = 2000  # сила двигателя (Н)
v_0 = 0  # начальная скорость (м/с)
k = 0.05  # коэффициент сопротивления (Н·с²/м²)

# Время расчета
t_max = m_f0 / dot_m_f  # время полного сгорания топлива
dt = 0.1  # шаг интегрирования (с)
t = np.arange(0, t_max + dt, dt)  # временной массив

# Инициализация переменных
v = np.zeros_like(t)  # скорость
m = m_o + m_f0 - dot_m_f * t  # масса объекта
a = np.zeros_like(t)  # ускорение
s = np.zeros_like(t)  # расстояние

# Расчёт данных
for i in range(1, len(t)):
    # Если топливо закончилось, скорость будет уменьшаться
    if m[i] > m_o:
        F_resistance = k * v[i-1]**2  # сопротивление от трения и аэродинамики
        a[i] = (F_engine - F_resistance) / m[i]  # ускорение
    else:
        F_resistance = k * v[i-1]**2  # сопротивление
        a[i] = -F_resistance / m[i]  # ускорение после окончания топлива (замедление)

    # Обновление скорости и расстояния
    v[i] = v[i-1] + a[i] * dt
    s[i] = s[i-1] + v[i] * dt

    # Остановка, если скорость стала отрицательной или объект остановился
    if v[i] < 0:
        v[i] = 0
        break

# Настройка графика
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, max(t))  # Ось времени
ax.set_ylim(0, max(v) * 1.1)  # Ось скорости
ax.set_xlabel("Время (с)")
ax.set_ylabel("Скорость (м/с)")
ax.grid()

# Линия скорости
line, = ax.plot([], [], 'b-', lw=2, label="Скорость (м/с)")
ax.legend()

# Текст для отображения текущего расстояния
distance_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

# Данные для анимации
x_data, y_data = [], []

# Функция обновления для анимации
def update(frame):
    # Остановка анимации, если скорость объекта стала равной нулю
    if v[frame] < 0:
        update.anim.event_source.stop()
        return line, distance_text

    # Обновление данных графика
    x_data.append(t[frame])  # Время на оси X
    y_data.append(v[frame])  # Скорость на оси Y
    line.set_data(x_data, y_data)

    # Обновление текста расстояния
    distance_text.set_text(f'Расстояние: {s[frame]:.2f} м')
    return line, distance_text

# Создание анимации
ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

# Отображение графика
plt.show()
