import os
import re
import tensorflow as tf
from fuzzywuzzy import process
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, AdamWeightDecay

# Wyłączanie ostrzeżeń
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Wczytywanie artykułu z pliku
with open("files/elektronika.txt", 'r', encoding='utf-8') as file:
    article = file.read()

# Inicjalizacja tokenizer i modelu
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Dane treningowe
train_data = [
    {
        "question": "Czym zajmuje się elektronika?",
        "answer_text": "Elektronika zajmuje się wytwarzaniem i przetwarzaniem sygnałów w postaci prądów, napięć elektrycznych oraz pól elektromagnetycznych."
    },
    {
        "question": "Jakie są podstawowe elementy elektroniczne?",
        "answer_text": "Podstawowe elementy elektroniczne to elementy aktywne (tranzystory, diody, układy scalone), bierne (rezystory, kondensatory, cewki) oraz optoelektroniczne i fotoniczne."
    },
    {
        "question": "Czym różni się elektronika od elektrotechniki?",
        "answer_text": "Elektronika zajmuje się przetwarzaniem sygnałów, a elektrotechnika głównie zagadnieniami związanymi z energią elektryczną."
    },
    {
        "question": "Kiedy nastąpiło wyodrębnienie elektroniki jako dziedziny nauki?",
        "answer_text": "Elektronika wyodrębniła się jako osobna dziedzina około 1906 roku, gdy Lee De Forest wynalazł triodę."
    },
    {
        "question": "Jakie są główne zastosowania elektroniki?",
        "answer_text": "Elektronika znajduje zastosowanie w telekomunikacji, inżynierii komputerowej, automatyce, medycynie, przemyśle oraz w technologiach mikrofalowych."
    },
    {
        "question": "Jakie są główne typy układów elektronicznych?",
        "answer_text": "Główne typy układów elektronicznych to układy analogowe (liniowe i nieliniowe) oraz cyfrowe (kombinacyjne i sekwencyjne)."
    },
    {
        "question": "Co to jest mikroelektronika?",
        "answer_text": "Mikroelektronika to dział elektroniki zajmujący się projektowaniem i produkcją układów scalonych oraz miniaturyzacją komponentów elektronicznych."
    },
    {
        "question": "Jakie są kluczowe wynalazki w historii elektroniki?",
        "answer_text": "Do kluczowych wynalazków w historii elektroniki należą trioda, tranzystor, układy scalone oraz technologie półprzewodnikowe."
    },
    {
        "question": "Jakie są różnice między układami analogowymi a cyfrowymi?",
        "answer_text": "Układy analogowe przetwarzają sygnały ciągłe, natomiast układy cyfrowe operują na sygnałach dyskretnych, reprezentowanych przez wartości binarne."
    },
    {
        "question": "Jakie są podstawowe elementy półprzewodnikowe?",
        "answer_text": "Podstawowe elementy półprzewodnikowe to diody, tranzystory, tyrystory oraz układy scalone."
    },
    {
        "question": "Czym zajmuje się radioelektronika?",
        "answer_text": "Radioelektronika obejmuje technologie związane z radiokomunikacją, systemami radarowymi oraz układami mikrofalowymi."
    },
    {
        "question": "Jakie są zastosowania elektroniki w medycynie?",
        "answer_text": "Elektronika medyczna obejmuje urządzenia diagnostyczne, aparaturę monitorującą oraz technologie stosowane w inżynierii biomedycznej."
    },
    {
        "question": "Co to jest kompatybilność elektromagnetyczna?",
        "answer_text": "Kompatybilność elektromagnetyczna to zdolność urządzeń elektronicznych do poprawnego działania w otoczeniu pełnym różnych sygnałów elektromagnetycznych."
    }
]

# 🔍 Znajdowanie najlepszego fragmentu tekstu
def find_best_passage(answer_text, article):
    sentences = re.split(r'(?<=[.!?]) +', article)
    best_match, score = process.extractOne(answer_text, sentences)

    if score < 50:
        print(f"⚠️ Słabe dopasowanie ({score}%) dla: {answer_text}")
        return None

    print(f"✅ Dopasowanie ({score}%) dla: {answer_text} -> {best_match[:100]}...")
    return best_match

# 🔍 Znajdowanie indeksów odpowiedzi (ulepszone!)
def find_answer_positions(answer_text, best_sentence):
    start_idx = best_sentence.find(answer_text)

    if start_idx != -1:
        end_idx = start_idx + len(answer_text)
        return start_idx, end_idx
    
    # 💡 Jeśli dokładne dopasowanie się nie powiodło, szukamy podobnego fragmentu
    words = answer_text.split()
    for word in words:
        if word in best_sentence:
            start_idx = best_sentence.find(word)
            end_idx = start_idx + len(word)
            return start_idx, end_idx
    
    print(f"⚠️ Nie znaleziono dopasowania dla: {answer_text}")
    return 0, 1

# Przygotowanie danych treningowych
inputs = []
start_positions = []
end_positions = []

for entry in train_data:
    best_sentence = find_best_passage(entry["answer_text"], article)
    if not best_sentence:
        continue

    encoded = tokenizer(
        entry["question"], best_sentence,
        truncation=True, padding="max_length", return_tensors="tf"
    )

    start_idx, end_idx = find_answer_positions(entry["answer_text"], best_sentence)
    if start_idx is None or end_idx is None:
        continue

    inputs.append(encoded)
    start_positions.append(start_idx)
    end_positions.append(end_idx)

# Sprawdzenie, czy mamy dane do trenowania
if not inputs:
    raise ValueError("❌ Brak poprawnych danych wejściowych do treningu!")

input_ids = tf.concat([inp["input_ids"] for inp in inputs], axis=0)
attention_masks = tf.concat([inp["attention_mask"] for inp in inputs], axis=0)

# Kompilacja modelu
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

start_positions = tf.convert_to_tensor(start_positions, dtype=tf.int32)
end_positions = tf.convert_to_tensor(end_positions, dtype=tf.int32)

print(f"input_ids shape: {input_ids.shape}")
print(f"attention_masks shape: {attention_masks.shape}")
print(f"start_positions shape: {tf.convert_to_tensor(start_positions).shape}")
print(f"end_positions shape: {tf.convert_to_tensor(end_positions).shape}")

print(f"inputs_ids: {input_ids}")
print(f"attention_mask: {attention_masks}")
print(f"start_positions: {start_positions}")
print(f"end_positions: {end_positions}")

print(f"start position dtype: {start_positions.dtype}")
print(f"end position dtype: {end_positions.dtype}")

# Trenowanie modelu
model.fit(
    (input_ids, attention_masks),
    (start_positions, end_positions),
    validation_split=0.2,
    epochs=8,
    batch_size=4,
)

# Zapisywanie modelu
model.save_pretrained("saved_model/Kally")
tokenizer.save_pretrained("saved_model/Kally")
print("✅ Model Kally został zapisany!")
