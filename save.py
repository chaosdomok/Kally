import wikipedia

# Setting language to Polish
wikipedia.set_lang("pl")

# Downloading from Wikipedia
def save_article_to_file(topic, filename):
    try:
        page = wikipedia.page(topic)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(page.content)
        print(f"Zapisano artykuł na temat '{topic}' do pliku '{filename}'")
    except wikipedia.exceptions.PageError:
        print("❌ Nie znaleziono artykułu.")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"❌ Znaleziono kilka wyników: {e.options}")

# File saving
save_article_to_file("Elektronika", "files/elektronika.txt")
