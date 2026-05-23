import matplotlib.pyplot as plt
from factor_analysis1 import eigen_values

# gender_counts = df["Пол"].value_counts()
#
# plt.figure(figsize=(6,6))
# plt.pie(gender_counts,
#         labels=gender_counts.index,
#         autopct='%1.1f%%')
#
# plt.title("Распределение респондентов по полу")
# plt.show()
#
# #____________________________________________________________________________
# social_counts = df["Сколько времени в среднем вы проводите в социальных сетях?"].value_counts()
# order_social = [
#     "Менее 1 часа в день",
#     "1-2 часа в день",
#     "3-4 часа в день",
#     "5-6 часов в день",
#     "Более 6 часов в день"
# ]
# # переупорядочивание
# social_counts = social_counts.reindex(order_social)
#
# plt.figure(figsize=(8,5))
# plt.bar(social_counts.index, social_counts.values)
#
# plt.xticks(rotation=15)
# plt.xlabel("Время")
# plt.ylabel("Количество респондентов")
# plt.title("Время, проводимое в социальных сетях")
#
# plt.show()

#____________________________________________________________________________
# reading_counts = df["Сколько времени в среднем вы уделяете чтению " \
#               "книг?"].value_counts()
# order_reading = [
#     "Не читаю",
#     "Менее 1 часа в неделю",
#     "1-3 часа в неделю",
#     "4-7 часов в неделю",
#     "Более 7 часов в неделю"
# ]
# # переупорядочивание
# reading_counts = reading_counts.reindex(order_reading)
#
# plt.figure(figsize=(8,5))
# plt.bar(reading_counts.index, reading_counts.values)
#
# plt.xticks(rotation=15)
# plt.xlabel("Время")
# plt.ylabel("Количество респондентов")
# plt.title("Время, проводимое за чтением книг")
#
# plt.show()

# ________________факторный анализ объектов_____________

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(eigen_values) + 1),
    eigen_values,
    marker='o'
)

plt.axhline(y=1, linestyle='--')

plt.xlabel("Номер фактора")
plt.ylabel("Собственное значение")
plt.title("График каменистой осыпи (Scree Plot)")

plt.grid()

plt.show()