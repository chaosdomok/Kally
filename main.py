import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering

# Załaduj zapisany model i tokenizer
tokenizer = BertTokenizerFast.from_pretrained("kally/kally")
model = TFBertForQuestionAnswering.from_pretrained("kally/kally")

# Funkcja do odpowiadania na pytania
def answer_question(question, context):
    encoded = tokenizer(
        question,
        context,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="tf"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0] + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index])
    )
    
    return answer

# Wczytaj kontekst z pliku
with open("files/elektronika.txt", 'r', encoding='utf-8') as file:
    context = file.read()

while True:
    question = input("Podaj pytanie: ")
    answer = answer_question(question, context)
    print(answer)