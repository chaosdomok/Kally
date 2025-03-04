import wikipedia 

wikipedia.set_lang('pl')

def save_article_to_file(topic, filename):
    try:
        page = wikipedia.page(topic)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(page.content)
            print(f"zapisano atykuł na temat: '{topic}' do pliku '{filename}'")
    except wikipedia.exceptions.PageError:
        return "Nie znaleziono artykułu"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Znaleziono kilka wyników: {e.options}"


# Zapis artykułu w pliku

# Pytania
save_article_to_file('Elektronika', "files/elektronika.txt")