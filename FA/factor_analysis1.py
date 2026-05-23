from main import df, expert_matrix
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, \
    calculate_kmo, FactorAnalyzer
import pandas as pd

#====================================== Подготовка данных ===================
factor_matrix = expert_matrix.copy()    # копия матрицы для факторного анализа
factor_matrix.columns = [f"Q{i}" for i in range(1, 11)]
reverse_questions = ["Q5", "Q6", "Q7"]  # инверсия

factor_matrix[reverse_questions] = 6 - factor_matrix[reverse_questions]
corr_matrix = factor_matrix.corr()  # корреляц. матрица
corr_matrix.to_excel("tables/corr_matrix.xlsx")

factor_matrix_clean = factor_matrix.dropna()    #удаление строк с пустотами

#====================================== Проверка пригодности ================
# ------------------------------------------------критерий Бартлетта
hi_square_value, p_value = calculate_bartlett_sphericity(factor_matrix_clean)

# -------------------------------------------------КМО
kmo_all, kmo_model = calculate_kmo(factor_matrix_clean)

#====================================== Факторный анализ ===================
# создаем модель без ограничения числа факторов
fa = FactorAnalyzer(rotation=None)
# обучаем модель
fa.fit(factor_matrix_clean)

# получаем eigenvalues
eigen_values, vectors = fa.get_eigenvalues()

# расчёт факторных нагрузок
# создаем модель с 3 факторами
fa = FactorAnalyzer(
    n_factors=3,
    rotation='varimax'
)

# обучаем модель
fa.fit(factor_matrix_clean)

# получаем факторные нагрузки
loadings = pd.DataFrame(
    fa.loadings_,
    index=factor_matrix_clean.columns,
    columns=['Factor1', 'Factor2', 'Factor3']
)


if __name__ == "__main__":
    # print(corr_matrix)
    # print("Проверка критерия Бартлетта:")
    # print("хи-квадрат:", hi_square_value)
    # print("p-value:", p_value)
    # # print(factor_matrix.info())
    # # print(factor_matrix.isnull().sum())
    # print("Проверка КМО:")
    # print(kmo_model)
    # print(f"КМО для каждого: {kmo_all}")
    # print(eigen_values)
    print(loadings)


