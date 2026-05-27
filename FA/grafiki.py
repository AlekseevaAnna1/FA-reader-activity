import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Настройка шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'

print("Загрузка данных...")
df = pd.read_excel('tables/opros_data_soi3.xlsx', sheet_name='Sheet')
df = df.dropna(subset=['Я получаю удовольствие от чтения книг'])
df = df.reset_index(drop=True)
n = len(df)
print(f"Загружено {n} респондентов\n")

# Переименование для удобства
rename_dict = {
    'Я получаю удовольствие от чтения книг': 'Q1',
    'Я считаю чтение важным способом саморазвития': 'Q2',
    'Чтение помогает человеку развивать мышление и речь': 'Q3',
    'Мне легко концентрироваться на длинных текстах': 'Q4',
    'Социальные сети и короткий цифровой контент отвлекают меня от чтения книг': 'Q5',
    'Я предпочитаю короткий цифровой контент длинным текстам': 'Q6',
    'После длительного использования телефона мне сложнее читать книги': 'Q7',
    'В моей семье чтение считалось важной привычкой': 'Q8',
    'В детстве я часто видел(а), как близкие мне люди читают книги': 'Q9',
    'В моем кругу общения люди интересуются литературой и чтением': 'Q10'
}
df = df.rename(columns=rename_dict)

# Инверсия Q5, Q6, Q7
for col in ['Q5', 'Q6', 'Q7']:
    df[col] = 6 - df[col]

# Обработка пропусков в вопросах Q1-Q10 (замена на среднее)
questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
for col in questions:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"  {col}: пропуск заменён на среднее = {mean_val:.2f}")

question_labels_short = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
question_labels_long = ['Q1\nудовольствие', 'Q2\nсаморазвитие', 'Q3\nразвитие мышления',
                        'Q4\nконцентрация', 'Q5\nсоцсети\nне отвлекают',
                        'Q6\nне предпочитаю\nкороткий контент',
                        'Q7\nпосле телефона\nне сложнее', 'Q8\nсемья\nчитает',
                        'Q9\nдетство\nчтение', 'Q10\nокружение\nчитает']

# Расчёт статистик
means = [df[q].mean() for q in questions]
std_devs = [df[q].std() for q in questions]
cv = [(df[q].std() / df[q].mean()) * 100 for q in questions]

print("\nПодготовка данных для Scree plot...")

# Берём только данные без пропусков
data_for_pca = df[questions].copy()

# Ещё раз убеждаемся, что пропусков нет
if data_for_pca.isnull().any().any():
    print("  Обнаружены пропуски, заменяем на средние...")
    for col in questions:
        if data_for_pca[col].isnull().any():
            data_for_pca[col] = data_for_pca[col].fillna(data_for_pca[col].mean())

print(f"  Размер матрицы: {data_for_pca.shape[0]} x {data_for_pca.shape[1]}")

# Стандартизация
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_pca)

# PCA
pca = PCA()
pca.fit(data_scaled)
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"\nСобственные значения: {[round(x, 3) for x in eigenvalues]}")
print(f"Доля дисперсии: {[round(x, 3) for x in explained_variance_ratio]}")
print(f"Накопленная дисперсия: {[round(x, 3) for x in cumulative_variance]}")

print("\n" + "=" * 50)
print("Графики будут открываться по очереди.")
print("Закройте текущий график, чтобы увидеть следующий.")
print("=" * 50)

print("\n[График 1/7] Распределение по полу...")
gender_counts = df['Пол'].value_counts()

plt.figure(figsize=(8, 6))
colors = ['#FF69B4', '#4169E1']
plt.pie(gender_counts.values, labels=gender_counts.index, colors=colors,
        autopct='%1.0f%%', startangle=90, explode=(0.05, 0))
plt.title(f'Распределение респондентов по полу (n = {n})', fontsize=14)
plt.show()

print("\n[График 2/7] Распределение по возрасту...")
age_counts = df['Ваш возраст'].value_counts()

age_values = [
    age_counts.get('17-18', 0) + age_counts.get('17–18', 0),
    age_counts.get('19-20', 0) + age_counts.get('19–20', 0),
    age_counts.get('21-22', 0) + age_counts.get('21–22', 0),
    age_counts.get('23 года и старше', 0)
]
age_labels = ['17-18', '19-20', '21-22', '23+']

plt.figure(figsize=(8, 6))
bars = plt.bar(age_labels, age_values, color='skyblue', edgecolor='black', width=0.6)
plt.xlabel('Возрастная группа', fontsize=12)
plt.ylabel('Количество респондентов', fontsize=12)
plt.title(f'Распределение респондентов по возрасту (n = {n})', fontsize=14)

for bar, val in zip(bars, age_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontsize=11)
plt.show()

print("\n[График 3/7] Средние значения по вопросам...")

fig, ax = plt.subplots(figsize=(12, 6))
colors_mean = ['#2ecc71' if m >= 4 else '#f39c12' if m >= 3 else '#e74c3c' for m in means]
bars = ax.bar(question_labels_long, means, color=colors_mean, edgecolor='black', linewidth=1)

ax.axhline(y=3, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=4, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_ylabel('Средний балл (1-5)', fontsize=12)
ax.set_title(f'Средние значения ответов по вопросам (n={n})', fontsize=14)
ax.set_ylim(0, 5.5)

for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n[График 4/7] Коэффициент вариации (разброс мнений)...")

fig, ax = plt.subplots(figsize=(12, 6))
colors_cv = ['#27ae60' if c < 30 else '#e67e22' if c < 40 else '#e74c3c' for c in cv]
bars = ax.bar(question_labels_long, cv, color=colors_cv, edgecolor='black', linewidth=1)

ax.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=40, color='orange', linestyle='--', linewidth=1, alpha=0.7)

ax.set_ylabel('Коэффициент вариации (%)', fontsize=12)
ax.set_title(f'Разброс мнений респондентов (коэффициент вариации, n={n})', fontsize=14)
ax.set_ylim(0, 55)

for bar, c in zip(bars, cv):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{c:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n[График 5/7] Scree plot (собственные значения факторов)...")

fig, ax = plt.subplots(figsize=(10, 6))

# Собственные значения
x = np.arange(1, len(eigenvalues) + 1)
ax.plot(x, eigenvalues, 'bo-', linewidth=2, markersize=8, label='Собственные значения')

# Горизонтальная линия на уровне 1 (критерий Кайзера)
ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Критерий Кайзера (λ=1)')

# Закрашиваем область выше 1
ax.fill_between(x, eigenvalues, 1, where=(eigenvalues >= 1), alpha=0.2, color='green')

# Подписи
ax.set_xlabel('Номер фактора', fontsize=12)
ax.set_ylabel('Собственное значение (λ)', fontsize=12)
ax.set_title('График каменистой осыпи (Scree plot)', fontsize=14)

# Значения над точками
for i, val in enumerate(eigenvalues):
    ax.text(i + 1, val + 0.1, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Отметка для λ=1
ax.text(len(eigenvalues) + 0.5, 1.05, 'λ = 1', color='red', fontsize=10)

# Дополнительная информация на графике
ax.text(0.5, max(eigenvalues) * 0.9,
        f'λ₁ = {eigenvalues[0]:.3f} ({explained_variance_ratio[0]*100:.1f}% дисперсии)',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, max(eigenvalues) * 0.8,
        f'λ₂ = {eigenvalues[1]:.3f} ({explained_variance_ratio[1]*100:.1f}% дисперсии)',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, max(eigenvalues) * 0.7,
        f'λ₃ = {eigenvalues[2]:.3f} ({explained_variance_ratio[2]*100:.1f}% дисперсии)',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f'F{i}' for i in x])

plt.tight_layout()
plt.show()

print("\n[График 6/7] Распределение по времени в соцсетях...")

social_counts = df['Сколько времени в среднем вы проводите в социальных сетях?'].value_counts()

social_labels = ['< 1 ч', '1-2 ч', '3-4 ч', '5-6 ч', '> 6 ч']
social_values = [
    social_counts.get('Менее 1 часа в день', 0),
    social_counts.get('1-2 часа в день', 0) + social_counts.get('1–2 часа в день', 0),
    social_counts.get('3-4 часа в день', 0) + social_counts.get('3–4 часа в день', 0),
    social_counts.get('5-6 часов в день', 0) + social_counts.get('5–6 часов в день', 0),
    social_counts.get('Более 6 часов в день', 0)
]

plt.figure(figsize=(8, 6))
bars = plt.bar(social_labels, social_values, color='skyblue', edgecolor='black', width=0.6)
plt.xlabel('Время в социальных сетях в день', fontsize=12)
plt.ylabel('Количество респондентов', fontsize=12)
plt.title(f'Распределение времени в соцсетях (n={n})', fontsize=14)

for bar, val in zip(bars, social_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontsize=11)

if social_values[0] == 4:
    plt.annotate('всего 4 человека', xy=(0, 4), xytext=(0, 6),
                 ha='center', fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.show()

print("\n[График 7/7] Распределение по времени на чтение...")

reading_counts = df['Сколько времени в среднем вы уделяете чтению книг?'].value_counts()

reading_labels = ['Не читаю', '< 1 ч', '1-3 ч', '4-7 ч', '> 7 ч']
reading_values = [
    reading_counts.get('Не читаю', 0),
    reading_counts.get('Менее 1 часа в неделю', 0),
    reading_counts.get('1-3 часа в неделю', 0) + reading_counts.get('1–3 часа в неделю', 0),
    reading_counts.get('4-7 часов в неделю', 0) + reading_counts.get('4–7 часов в неделю', 0),
    reading_counts.get('Более 7 часов в неделю', 0)
]

plt.figure(figsize=(8, 6))
bars = plt.bar(reading_labels, reading_values, color='lightgreen', edgecolor='black', width=0.6)
plt.xlabel('Время на чтение в неделю', fontsize=12)
plt.ylabel('Количество респондентов', fontsize=12)
plt.title(f'Распределение времени на чтение (n={n})', fontsize=14)

for bar, val in zip(bars, reading_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontsize=11)

less_than_3 = reading_values[0] + reading_values[1] + reading_values[2]
pct_less_3 = round(less_than_3 / n * 100)
plt.annotate(f'Менее 3 часов: {less_than_3} чел. ({pct_less_3}%)',
             xy=(2, reading_values[2]), xytext=(2, max(reading_values) + 1.5),
             ha='center', fontsize=10, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("СТАТИСТИКА ПО ВОПРОСАМ")
print("=" * 50)

print(f"\n{'Вопрос':<10} {'Среднее':<10} {'Ст.откл.':<10} {'CV (%)':<10}")
print("-" * 40)
for i, q in enumerate(questions):
    print(f"{q:<10} {means[i]:.2f}{' ' * 8} {std_devs[i]:.2f}{' ' * 8} {cv[i]:.1f}")

print(f"\n{'=' * 50}")
print(f"Всего респондентов: {n}")
print(f"Максимальное среднее: Q{questions[np.argmax(means)]} = {max(means):.2f}")
print(f"Минимальное среднее: Q{questions[np.argmin(means)]} = {min(means):.2f}")
print(f"Максимальный разброс (CV): Q{questions[np.argmax(cv)]} = {max(cv):.1f}%")
print(f"Минимальный разброс (CV): Q{questions[np.argmin(cv)]} = {min(cv):.1f}%")

print("\n" + "=" * 50)
print("SCREE PLOT (собственные значения):")
for i, val in enumerate(eigenvalues):
    print(f"  F{i+1}: {val:.3f} ({explained_variance_ratio[i]*100:.1f}%)")
print("=" * 50)