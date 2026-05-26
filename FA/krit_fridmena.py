import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import friedmanchisquare
import os

file_path = "FA/tables/expert_matrix.xlsx"

df_raw = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)

print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)
print(f"Исходная размерность: {df_raw.shape[0]} строк x {df_raw.shape[1]} столбцов")
print("\nПервые 5 строк исходных данных:")
print(df_raw.head())

df_clean = df_raw.dropna(how='all')
print(f"\nПосле удаления пустых строк: {df_clean.shape[0]} экспертов")
print(f"Пропуски в каждой колонке:\n{df_clean.isnull().sum()}")

for col in df_clean.columns:
    if df_clean[col].isnull().any():
        mean_val = df_clean[col].mean()
        df_clean[col] = df_clean[col].fillna(mean_val)
        print(f"  Пропуски в '{col}' заполнены средним = {mean_val:.2f}")

column_mapping = {
    df_clean.columns[0]: 'Q1',
    df_clean.columns[1]: 'Q2',
    df_clean.columns[2]: 'Q3',
    df_clean.columns[3]: 'Q4',
    df_clean.columns[4]: 'Q5',
    df_clean.columns[5]: 'Q6',
    df_clean.columns[6]: 'Q7',
    df_clean.columns[7]: 'Q8',
    df_clean.columns[8]: 'Q9',
    df_clean.columns[9]: 'Q10',
}
df = df_clean.rename(columns=column_mapping)

print(f"\nПереименованные столбцы: {list(df.columns)}")
print(f"\nИтоговая матрица: {df.shape[0]} экспертов x {df.shape[1]} вопросов")

print("\n" + "=" * 70)
print("ОБРАТНОЕ КОДИРОВАНИЕ (Q5, Q6, Q7)")
print("=" * 70)

print("До обратного кодирования (больше = больше отвлечения):")
for col in ['Q5', 'Q6', 'Q7']:
    print(f"  {col}: среднее = {df[col].mean():.2f}, мин = {df[col].min()}, макс = {df[col].max()}")

for col in ['Q5', 'Q6', 'Q7']:
    df[col] = 6 - df[col]

print("\nПосле обратного кодирования (больше = меньше отвлечения, лучше чтение):")
for col in ['Q5', 'Q6', 'Q7']:
    print(f"  {col}: среднее = {df[col].mean():.2f}, мин = {df[col].min()}, макс = {df[col].max()}")

def kendalls_w_with_friedman(df):
    n = df.shape[0]
    k = df.shape[1]

    ranked = df.rank(axis=1, method='average')
    R_j = ranked.sum(axis=0).values
    R_mean = n * (k + 1) / 2
    S = np.sum((R_j - R_mean) ** 2)

    T_sum = 0
    for i in range(n):
        value_counts = df.iloc[i, :].value_counts()
        for t in value_counts.values:
            if t > 1:
                T_sum += (t ** 3 - t)

    denominator = (n ** 2 * (k ** 3 - k)) - (n * T_sum)
    W = (12 * S) / denominator

    chi2_friedman = (12 * S) / (n * k * (k + 1) - (T_sum / (k - 1)))
    p_value = 1 - chi2.cdf(chi2_friedman, df=k - 1)
    chi2_critical = chi2.ppf(0.95, df=k - 1)

    if W < 0.2:
        w_int = "очень слабая"
    elif W < 0.4:
        w_int = "слабая"
    elif W < 0.6:
        w_int = "умеренная"
    elif W < 0.8:
        w_int = "сильная"
    else:
        w_int = "очень сильная"

    return {
        'n': n,
        'k': k,
        'S': S,
        'T_sum': T_sum,
        'W': W,
        'W_interpretation': w_int,
        'chi2': chi2_friedman,
        'chi2_critical': chi2_critical,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'ranked_matrix': ranked,
        'R_j': R_j,
        'R_mean': R_mean
    }

print("\n" + "=" * 70)
print("ПУНКТ 6: ОЦЕНКА СОГЛАСОВАННОСТИ ЭКСПЕРТОВ")
print("=" * 70)

results = kendalls_w_with_friedman(df)

print(f"\nОСНОВНЫЕ ПОКАЗАТЕЛИ:")
print(f"   Число экспертов (n):          {results['n']}")
print(f"   Число вопросов (k):           {results['k']}")
print(f"   Сумма квадратов отклонений S:  {results['S']:.2f}")
print(f"   Поправка на связи SUM(T):      {results['T_sum']:.2f}")

print(f"\nКОЭФФИЦИЕНТ КОНКОРДАЦИИ КЕНДАЛЛА:")
print(f"   W = {results['W']:.4f}")
print(f"   Интерпретация: {results['W_interpretation']} согласованность")

print(f"\nКРИТЕРИЙ ФРИДМАНА:")
print(f"   chi2 (расчётное) = {results['chi2']:.2f}")
print(f"   chi2 (критическое, alpha=0.05, df={results['k'] - 1}) = {results['chi2_critical']:.2f}")
print(f"   p-value = {results['p_value']:.2e}")


print(f"\nСТАТИСТИЧЕСКИЙ ВЫВОД:")
if results['significant']:
    print(f"   Поскольку p-value = {results['p_value']:.2e} < 0.05,")
    print(f"   нулевая гипотеза об отсутствии согласованности ОТВЕРГАЕТСЯ.")
    print(f"   Согласованность экспертов является статистически значимой.")
else:
    print(f"   Поскольку p-value = {results['p_value']:.3f} >= 0.05,")
    print(f"   нулевая гипотеза об отсутствии согласованности НЕ ОТВЕРГАЕТСЯ.")

print(f"\nСУММЫ РАНГОВ ПО ВОПРОСАМ:")
R_j_df = pd.DataFrame({
    'Вопрос': [f'Q{i + 1}' for i in range(results['k'])],
    'Сумма рангов (R_j)': results['R_j'],
    'Отклонение от среднего': results['R_j'] - results['R_mean']
})
print(R_j_df.to_string(index=False))

print("\n" + "=" * 70)
print("ПУНКТ 7: ПЕРЕСЧЁТ ПОСЛЕ УДАЛЕНИЯ ПЛОХИХ ЭКСПЕРТОВ")
print("=" * 70)

expert_means = df.mean(axis=1)
overall_mean = expert_means.mean()
deviations = abs(expert_means - overall_mean)

print(f"\nСРЕДНИЕ БАЛЛЫ ЭКСПЕРТОВ:")
print(f"   Общее среднее по выборке: {overall_mean:.3f}")
print(f"   Стандартное отклонение средних: {expert_means.std():.3f}")
print(f"   Минимальный средний балл: {expert_means.min():.3f} (эксперт {expert_means.idxmin()})")
print(f"   Максимальный средний балл: {expert_means.max():.3f} (эксперт {expert_means.idxmax()})")

deviation_df = pd.DataFrame({
    'эксперт (исходный индекс)': expert_means.index,
    'средний балл': expert_means.values,
    'отклонение от общего среднего': deviations.values
}).sort_values('отклонение от общего среднего', ascending=False)

print("\nТОП-5 ЭКСПЕРТОВ С НАИБОЛЬШИМ ОТКЛОНЕНИЕМ:")
print(deviation_df.head(5).to_string(index=False))

n_remove = 5
to_remove = deviation_df.head(n_remove)['эксперт (исходный индекс)'].tolist()

df_cleaned = df.drop(index=to_remove)

print(f"\nУДАЛЕНЫ ЭКСПЕРТЫ с индексами: {to_remove}")
print(f"   Осталось экспертов: {df_cleaned.shape[0]}")

results_cleaned = kendalls_w_with_friedman(df_cleaned)

print(f"\nРЕЗУЛЬТАТЫ ПОСЛЕ УДАЛЕНИЯ:")
print(f"   W (конкордация) = {results_cleaned['W']:.4f} ({results_cleaned['W_interpretation']} согласованность)")
print(f"   chi2 Фридмана = {results_cleaned['chi2']:.2f}")
print(f"   p-value = {results_cleaned['p_value']:.2e}")

print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ (ДО vs ПОСЛЕ УДАЛЕНИЯ)")
print("=" * 70)

w_change = (results_cleaned['W'] - results['W']) / results['W'] * 100
chi2_change = (results_cleaned['chi2'] - results['chi2']) / results['chi2'] * 100

print(f"\n{'Показатель':<30} {'ДО удаления':<18} {'ПОСЛЕ удаления':<18} {'Изменение':<12}")
print("-" * 78)
print(f"{'Коэф. конкордации W':<30} {results['W']:<18.4f} {results_cleaned['W']:<18.4f} {w_change:>+10.2f}%")
print(f"{'chi2 Фридмана':<30} {results['chi2']:<18.2f} {results_cleaned['chi2']:<18.2f} {chi2_change:>+10.2f}%")
print(f"{'p-value':<30} {results['p_value']:<18.2e} {results_cleaned['p_value']:<18.2e} {'---':<12}")

print(f"\nВЫВОД ПО ПУНКТУ 7:")
if w_change > 0:
    print(f"   Удаление плохих экспертов привело к УВЕЛИЧЕНИЮ коэффициента")
    print(f"   конкордации на {w_change:.1f}%, что подтверждает, что эти эксперты")
    print(f"   действительно снижали общую согласованность.")
else:
    print(f"   Удаление плохих экспертов привело к УМЕНЬШЕНИЮ коэффициента")
    print(f"   конкордации на {abs(w_change):.1f}%.")

print("\n" + "=" * 70)
print("ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА (scipy.stats.friedmanchisquare)")
print("=" * 70)

columns_list = [df[col].values for col in df.columns]
chi2_scipy, p_scipy = friedmanchisquare(*columns_list)

print(f"chi2 Фридмана (scipy):  {chi2_scipy:.2f}")
print(f"p-value (scipy):       {p_scipy:.2e}")

print(f"\nСравнение с нашей реализацией:")
print(f"  Наша chi2:     {results['chi2']:.2f}")
print(f"  Scipy chi2:    {chi2_scipy:.2f}")
print(f"  Разница:       {abs(results['chi2'] - chi2_scipy):.2f} ({abs(results['chi2'] / chi2_scipy - 1) * 100:.1f}%)")
