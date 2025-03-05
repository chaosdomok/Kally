# 📌 Wczytujemy zapisany model
tokenizer = BertTokenizer.from_pretrained("saved_model/Kally")
model = TFBertForQuestionAnswering.from_pretrained("saved_model/Kally")
print("✅ Model Kally został wczytany!")

# 📌 Testowanie modelu Kally
def ask_question(question, article):
    inputs = tokenizer(question, article, return_tensors="tf")
    outputs = model(inputs)

    start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]
    answer = tokenizer.decode(inputs["input_ids"][0][start:end+1])
    return answer

# 🔹 Przykładowe pytanie
question = "Kto stworzył termin Fizyka Kwantowa?"
answer = ask_question(question, article)
print(f"❓ Pytanie: {question}\n✅ Odpowiedź: {answer}")