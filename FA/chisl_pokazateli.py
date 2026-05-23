from main import expert_matrix
import pandas as pd

stats_table = pd.DataFrame({
    "Mean": expert_matrix.mean(),
    "Median": expert_matrix.median(),
    "Mode": expert_matrix.mode().iloc[0],
    "Variance": expert_matrix.var(),
    "Std": expert_matrix.std(),
    "Min": expert_matrix.min(),
    "Max": expert_matrix.max(),
    "CV (%)": (expert_matrix.std() / expert_matrix.mean()) * 100
})
stats_table.index = [f"Q{i}" for i in range(1, 11)]
if __name__ == "__main__":
    stats_table.to_excel("tables/stats_table.xlsx")
    print(stats_table)
