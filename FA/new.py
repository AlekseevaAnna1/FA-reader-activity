
import pandas as pd
import numpy as np
from scipy.stats import chi2, f
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("ПРАКТИЧЕСКАЯ РАБОТА №3 — ДОПОЛНИТЕЛЬНАЯ ЧАСТЬ")
print("=" * 70)

print("\n1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("-" * 50)

# Загрузка файла
df_raw = pd.read_excel(r'C:\FA-reader-activity\FA\tables\opros_data_soi3.xlsx', sheet_name='Sheet')
print(f"Загружено {len(df_raw)} строк")

# Переименование столбцов
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

df_raw = df_raw.rename(columns=rename_dict)

# Выделение основных вопросов
questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
df_main = df_raw[questions].copy()

# Удаление пустых строк
df_main = df_main.dropna(subset=questions, how='all')
df_main = df_main.dropna(subset=['Q1'])
df_main = df_main.reset_index(drop=True)
print(f"Респондентов после очистки: {len(df_main)}")

# Обработка пропусков (замена средним)
for col in questions:
    if df_main[col].isnull().any():
        mean_val = df_main[col].mean()
        df_main[col] = df_main[col].fillna(mean_val)
        print(f"  {col}: пропуск заменён на {mean_val:.2f}")

# Обратное кодирование Q5, Q6, Q7
for col in ['Q5', 'Q6', 'Q7']:
    df_main[col] = 6 - df_main[col]
print("  Q5, Q6, Q7 инвертированы")

# Добавление профильных характеристик
profile_cols = ['Пол', 'Ваш возраст',
                'Сколько времени в среднем вы проводите в социальных сетях?',
                'Сколько времени в среднем вы уделяете чтению книг?']

for col in profile_cols:
    if col in df_raw.columns:
        df_main[col] = df_raw[col]

# Кодирование категориальных переменных
df_main['Пол_код'] = df_main['Пол'].map({'Мужской': 0, 'Женский': 1})

age_map = {'17-18': 1, '17–18': 1, '19-20': 2, '19–20': 2,
           '21-22': 3, '21–22': 3, '23 года и старше': 4}
df_main['Возраст_код'] = df_main['Ваш возраст'].map(age_map)

social_map = {'Менее 1 часа в день': 1, '1-2 часа в день': 2, '1–2 часа в день': 2,
              '3-4 часа в день': 3, '3–4 часа в день': 3,
              '5-6 часов в день': 4, '5–6 часов в день': 4,
              'Более 6 часов в день': 5}
df_main['Соцсети_код'] = df_main['Сколько времени в среднем вы проводите в социальных сетях?'].map(social_map)

reading_map = {'Не читаю': 1, 'Менее 1 часа в неделю': 2,
               '1-3 часа в неделю': 3, '1–3 часа в неделю': 3,
               '4-7 часов в неделю': 4, '4–7 часов в неделю': 4,
               'Более 7 часов в неделю': 5}
df_main['Чтение_код'] = df_main['Сколько времени в среднем вы уделяете чтению книг?'].map(reading_map)

# Удаление строк с пропусками в кодированных переменных
df_model = df_main.dropna(subset=['Пол_код', 'Возраст_код', 'Соцсети_код', 'Чтение_код'])
print(f"После кодирования: {len(df_model)} респондентов")

print("\n" + "=" * 70)
print("2. РЕГРЕССИОННЫЙ АНАЛИЗ (зависимая переменная — Q7)")
print("-" * 50)

# Подготовка данных
Y = df_model['Q7'].values
X = df_model[['Пол_код', 'Возраст_код', 'Соцсети_код', 'Чтение_код']].values

# Добавляем столбец единиц для константы
X_with_const = np.column_stack([np.ones(len(X)), X])

# Расчёт коэффициентов методом наименьших квадратов: β = (X^T X)^(-1) X^T Y
XtX = X_with_const.T @ X_with_const
XtX_inv = np.linalg.inv(XtX)
beta = XtX_inv @ (X_with_const.T @ Y)

# Расчёт предсказанных значений и остатков
Y_pred = X_with_const @ beta
residuals = Y - Y_pred

# R² и скорректированный R²
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
n = len(Y)
k = X_with_const.shape[1]
r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k)

# Стандартные ошибки, t-статистики, p-values (ПРАВИЛЬНЫЕ)
mse = ss_res / (n - k)  # Mean squared error
var_beta = mse * np.diag(XtX_inv)
se_beta = np.sqrt(var_beta)
t_stats = beta / se_beta

# ПРАВИЛЬНЫЙ расчёт p-value (используем распределение Стьюдента)
from scipy.stats import t as t_dist
p_values = [2 * (1 - t_dist.cdf(np.abs(t), df=n - k)) for t in t_stats]

# F-статистика и её p-value
ss_reg = ss_tot - ss_res
f_stat = (ss_reg / (k - 1)) / (ss_res / (n - k))
f_pvalue = 1 - f.cdf(f_stat, k - 1, n - k)

# Вывод результатов
print(f"\nКоэффициенты регрессии:")
print(f"  Константа (β0): {beta[0]:.4f} (p = {p_values[0]:.4f})")
print(f"  Пол (β1): {beta[1]:.4f} (p = {p_values[1]:.4f})")
print(f"  Возраст (β2): {beta[2]:.4f} (p = {p_values[2]:.4f})")
print(f"  Время в соцсетях (β3): {beta[3]:.4f} (p = {p_values[3]:.4f})")
print(f"  Время на чтение (β4): {beta[4]:.4f} (p = {p_values[4]:.4f})")

print(f"\nПоказатели качества модели:")
print(f"  R² = {r_squared:.3f}")
print(f"  Скорректированный R² = {r_squared_adj:.3f}")
print(f"  F-статистика = {f_stat:.2f}, p = {f_pvalue:.4f}")

if f_pvalue < 0.05:
    print("  → Модель статистически значима")
else:
    print("  → Модель статистически не значима")

print("\n" + "=" * 70)
print("3. ПЕРЕСЧЁТ СОГЛАСОВАННОСТИ ПОСЛЕ УДАЛЕНИЯ «ПЛОХИХ» ЭКСПЕРТОВ")
print("-" * 50)

data_matrix = df_model[questions].values
n, k = data_matrix.shape
print(f"Исходная матрица: {n} экспертов, {k} вопросов")


def friedman_kendall(matrix):
    n, k = matrix.shape
    ranks = np.zeros_like(matrix, dtype=float)

    # Ранжирование с учётом связей
    for i in range(n):
        row = matrix[i, :]
        sorted_indices = np.argsort(row)
        ranks[i, sorted_indices] = np.arange(1, k + 1)
        unique_vals = np.unique(row)
        for val in unique_vals:
            indices = np.where(row == val)[0]
            if len(indices) > 1:
                ranks[i, indices] = np.mean(ranks[i, indices])

    Rj = np.sum(ranks, axis=0)
    R_avg = n * (k + 1) / 2
    S = np.sum((Rj - R_avg) ** 2)
    W = 12 * S / (n ** 2 * (k ** 3 - k))
    chi2_stat = 12 * S / (n * k * (k + 1))
    p_value = 1 - chi2.cdf(chi2_stat, df=k - 1)

    return W, chi2_stat, p_value, S


# Исходная выборка
W_orig, chi2_orig, p_orig, S_orig = friedman_kendall(data_matrix)
print(f"\nИсходная выборка (n = {n}):")
print(f"  S = {S_orig:.2f}")
print(f"  W = {W_orig:.3f}")
print(f"  χ² = {chi2_orig:.2f}")
print(f"  p = {p_orig:.2e}")

# Вычисляем средний балл каждого эксперта
mean_scores = np.mean(data_matrix, axis=1)

# Общее среднее
overall_mean = np.mean(mean_scores)

# Отклонения
deviations = np.abs(mean_scores - overall_mean)

# Находим 5 экспертов с максимальным отклонением
worst_indices = np.argsort(deviations)[-5:]  # 5 экспертов с наибольшим отклонением
print(f"\n«Плохие» эксперты (максимальное отклонение среднего балла):")
for idx in worst_indices:
    print(f"  Эксперт {idx}: средний балл = {mean_scores[idx]:.2f}, отклонение = {deviations[idx]:.3f}")

# Удаление «плохих» экспертов
data_matrix_cleaned = np.delete(data_matrix, worst_indices, axis=0)
n_clean = data_matrix_cleaned.shape[0]

# Очищенная выборка
W_clean, chi2_clean, p_clean, S_clean = friedman_kendall(data_matrix_cleaned)
print(f"\nОчищенная выборка (n = {n_clean}):")
print(f"  S = {S_clean:.2f}")
print(f"  W = {W_clean:.3f}")
print(f"  χ² = {chi2_clean:.2f}")
print(f"  p = {p_clean:.2e}")

# Сравнение
print(f"\nИзменение W: {W_orig:.3f} → {W_clean:.3f} (+{(W_clean - W_orig) * 100:.1f}%)")

print("\n" + "=" * 70)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 70)

print("\n2.1 Регрессионный анализ:")
print(f"    Модель: Q7 = {beta[0]:.2f} + {beta[1]:.2f}·Пол + {beta[2]:.2f}·Возраст + {beta[3]:.2f}·Соцсети + {beta[4]:.2f}·Чтение")
print(f"    R² = {r_squared:.3f}, R²_adj = {r_squared_adj:.3f}")
print(f"    F = {f_stat:.2f}, p = {f_pvalue:.4f}")

print("\n2.2 Смежный опрос:")
print("    Разработано 10 вопросов по 3 блокам")

print("\n2.3 Предобработка:")
print("    Выполнены: инверсия Q5-Q7, замена пропусков средним, кодирование категорий")

print("\n2.4 Пересчёт согласованности:")
print(f"    W исходный = {W_orig:.3f} → W после удаления = {W_clean:.3f}")
print(f"    Изменение: +{(W_clean - W_orig) * 100:.1f}%")

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)