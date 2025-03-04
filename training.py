import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering
import re

# Inicjalizacja tokenizera i modelu
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Wczytanie tekstu z pliku
with open("files/elektronika.txt", 'r', encoding='utf-8') as file:
    context = file.read()

# Dane treningowe (przykładowe pytania i odpowiedzi)
train_data = [
    {"question": "Czym zajmuje się elektronika?", "answer_text": "wytwarzaniem i przetwarzaniem sygnałów w postaci prądów i napięć elektrycznych lub pól elektromagnetycznych"},
    {"question": "Kto wynalazł triodę?", "answer_text": "Lee De Forest"},
    {"question": "Jakie są typy układów elektronicznych?", "answer_text": "układy elektroniczne analogowe, układy cyfrowe kombinacyjne, układy cyfrowe sekwencyjne"},
    {"question": "Jakie elementy są używane w konstrukcji urządzeń elektronicznych?", "answer_text": "elementy aktywne, elementy bierne, elementy akustoelektroniczne, elementy optoelektroniczne, elementy fotoniczne"},
    {"question": "Co to jest mikroelektronika?", "answer_text": "mikroelektronika, zob. też nanoelektronika"},
    {"question": "Czym jest kompatybilność elektromagnetyczna?", "answer_text": "kompatybilność elektromagnetyczna"},
    {"question": "Kto wynalazł triodę i w którym roku?", "answer_text": "Lee De Forest wynalazł triodę około 1906 roku"},
    {"question": "Jakie dziedziny zawdzięczają rozwój elektronice?", "answer_text": "fizyce i matematyce"},
    {"question": "Jakie są elementy aktywne używane w elektronice?", "answer_text": "półprzewodnikowe (tranzystory, tyrystory, układy scalone, diody półprzewodnikowe itp.), lampy próżniowe (diody, triody, pentody itd.)"},
    {"question": "Jakie są elementy bierne używane w elektronice?", "answer_text": "rezystory, kondensatory, cewki"},
    {"question": "Jakie są elementy optoelektroniczne?", "answer_text": "lasery, światłowody, detektory promieniowania"}
]

def prepare_training_data(data, context):
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []

    for entry in data:
        question = entry["question"]
        answer_text = entry["answer_text"]

        encoded = tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="tf",
            return_offsets_mapping=True
        )

        offsets = encoded.pop("offset_mapping")[0]

        # Elastyczne wyszukiwanie odpowiedzi
        answer_pattern = re.compile(re.escape(answer_text.lower()), re.IGNORECASE)
        match = answer_pattern.search(context.lower())

        if match:
            answer_start_char = match.start()
            answer_end_char = match.end()
            print(f"Znaleziono odpowiedź '{answer_text}' w kontekście na pozycjach {answer_start_char} - {answer_end_char}.")
        else:
            print(f"Odpowiedź '{answer_text}' nie znaleziona w kontekście dla pytania: '{question}'")
            continue

        token_start_index = None
        token_end_index = None

        for idx, (start_offset, end_offset) in enumerate(offsets):
            if start_offset <= answer_start_char < end_offset:
                token_start_index = idx
            if start_offset < answer_end_char <= end_offset:
                token_end_index = idx
                break

        if token_start_index is None or token_end_index is None:
            print(f"Nie udało się znaleźć tokenów odpowiadających odpowiedzi dla pytania: '{question}'")
            print(f"Token start index: {token_start_index}, Token end index: {token_end_index}")
            continue

        input_ids.append(encoded["input_ids"][0])
        attention_masks.append(encoded["attention_mask"][0])
        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    print(f"Liczba przygotowanych przykładów: {len(input_ids)}")
    return input_ids, attention_masks, start_positions, end_positions

# Przygotowanie danych treningowych
input_ids, attention_masks, start_positions, end_positions = prepare_training_data(train_data, context)

if len(input_ids) == 0:
    print("Lista input_ids jest pusta! Upewnij się, że odpowiedzi znajdują się w kontekście.")
    exit()

# Konwersja danych do TensorFlow
input_ids_tensor = tf.stack(input_ids)
attention_masks_tensor = tf.stack(attention_masks)
start_positions_tensor = tf.convert_to_tensor(start_positions)
end_positions_tensor = tf.convert_to_tensor(end_positions)

# Kompilacja modelu
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer)

# Trening modelu
model.fit(
    [input_ids_tensor, attention_masks_tensor],
    [start_positions_tensor, end_positions_tensor],
    epochs=3,
    batch_size=2
)

# Zapis modelu
model.save_pretrained("kally/kally")
tokenizer.save_pretrained("kally/kally")

print("Model został zapisany!")
