from main import df
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, \
    calculate_kmo, FactorAnalyzer
import pandas as pd
#=======================================Подготовка данных====================
# извлечение данных
expert_features = df[[
    "Сколько времени в среднем вы проводите в социальных сетях?",
    "Сколько времени в среднем вы уделяете чтению книг?",
    "Насколько хорошо вы знакомы с современной культурой чтения среди молодежи?",
    "Насколько уверенно вы отвечали на вопросы данной анкеты?",
    "Насколько объективными вы считаете свои ответы?",
    "Насколько тема чтения и саморазвития близка лично вам?",
    "Сколько книг вы в среднем читаете в год?",
    "Как часто вы обсуждаете книги с друзьями или знакомыми?",
    "Как часто вы смотрите короткие видео? (TikTok, Reels, Shorts)?",
]].copy()
# переименование столбцов на короткие понятные названия
expert_features.columns = [
    "social_media",
    "reading_time",
    "reading_culture",
    "confidence",
    "objectivity",
    "topic_closeness",
    "books_per_year",
    "book_discussions",
    "short_videos",
]

# приведение текстовых ответов к шкале 1-5
social_media_map = {
    "Менее 1 часа в день": 1,
    "1-2 часа в день": 2,
    "3-4 часа в день": 3,
    "5-6 часов в день": 4,
    "Более 6 часов в день": 5
}
reading_time_map = {
    "Не читаю": 1,
    "Менее 1 часа в неделю": 2,
    "1-3 часа в неделю": 3,
    "4-7 часов в неделю": 4,
    "Более 7 часов в неделю": 5
}
books_map = {
    "0": 1,
    "1-2 ": 2,
    "3-5": 3,
    "6-10": 4,
    "более 10": 5
}
discussion_map = {
    "Никогда": 1,
    "Редко": 2,
    "Иногда": 3,
    "Часто": 4,
    "Очень часто": 5
}
videos_map = {
    "Не смотрю": 1,
    "Меньше 30 минут": 1,

    "30-60 минут": 2,

    "1-2 часа в день": 3,

    "3-4 часа в день": 4,

    "Более 4-6 часов в день": 5,
    "Более 6 часов в день": 5
}

# применение мэппинга
expert_features["social_media"] = expert_features["social_media"].map(
    social_media_map)

expert_features["reading_time"] = expert_features["reading_time"].map(
    reading_time_map)

expert_features["books_per_year"] = expert_features["books_per_year"].map(
    books_map)

expert_features["book_discussions"] = expert_features["book_discussions"].map(
    discussion_map)

expert_features["short_videos"] = expert_features["short_videos"].map(
    videos_map)

# reverse coding
reverse_cols = [
    "social_media",
    "short_videos",
]
expert_features = expert_features.dropna(how='all')  # удаление пропусков
expert_features[reverse_cols] = 6 - expert_features[reverse_cols]
expert_features = expert_features.fillna(
    expert_features.mean()
)
expert_features.to_excel("tables/expert_features.xlsx")

if __name__ == "__main__":
    print(expert_features)
