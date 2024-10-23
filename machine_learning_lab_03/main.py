import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re


# Завантажити стоп-слова для англійської
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


def process_text(text):
    # Переводимо в нижній регістр
    text = text.lower()
    # Видаляємо неалфавітні символи та пунктуацію
    text = re.sub(r'[^a-z\s]', '', text)
    # Токенізація тексту
    words = word_tokenize(text)
    # Видаляємо стоп-слова
    words = [word for word in words if word not in stop_words]
    return words


# Приклад використання
with open('lab3.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

processed_words = process_text(raw_text)
print(processed_words[:30])  # Перші 20 слів після обробки

# Розділяємо текст на глави за заголовками (CHAPTER + номер)
chapters = re.split(r'CHAPTER [IVXLCDM]+\.', raw_text)

# Видаляємо зайві пробіли та порожні рядки
chapters = [chapter.strip() for chapter in chapters if chapter.strip()]

# Перевіряємо кількість глав
print(f"Кількість глав: {len(chapters)}")

# Виводимо назви перших кількох глав як перевірку
for i, chapter in enumerate(chapters[:3], start=1):
    print(f"Глава {i}:")
    print(chapter[:100], '...\n')


vectorizer = TfidfVectorizer(max_features=20, stop_words='english')

# Обчислюємо TF-IDF матрицю для всіх глав
tfidf_matrix = vectorizer.fit_transform(chapters)

# Отримуємо список слів з векторизатора
feature_names = vectorizer.get_feature_names_out()

# Виведення Топ-20 слів для кожної глави
for i, chapter in enumerate(chapters):
    print(f"\nГлава {i + 1}:")
    # Вибираємо значення TF-IDF для поточної глави
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    # Отримуємо індекси Топ-20 слів за значенням TF-IDF
    top_indices = tfidf_scores.argsort()[-20:][::-1]
    # Виводимо Топ-20 слів з їхніми значеннями TF-IDF
    for idx in top_indices:
        print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")

# Створюємо LDA-модель
lda = LatentDirichletAllocation(n_components=3, random_state=42)  # Приклад: 3 теми, 5 забагато, є 3 однакові теми, а це погано
lda.fit(tfidf_matrix)

# Виводимо Топ-10 слів, для кожної теми
for i, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    topic_words = [vectorizer.get_feature_names_out()[idx] for idx in top_words_idx]
    print(f"Тема {i + 1}: {', '.join(topic_words)}")