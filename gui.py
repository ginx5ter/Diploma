import customtkinter as ctk
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
import random
import string

model = keras.models.load_model('best_transformer_model.h5')

VOCAB_SIZE = 100
SEQUENCE_LEN = 50
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token='UNK')
data = pd.read_csv('data.csv')
tokenizer.fit_on_texts([str(password) for password in data['password'].values])

def load_common_words(filename):
    with open(filename, 'r') as file:
        common_words = [word.strip().lower() for word in file.readlines()]
    return common_words

def load_common_sequences(filename):
    with open(filename, 'r') as file:
        common_sequences = [word.strip().lower() for word in file.readlines()]
    return common_sequences

common_words = load_common_words('words.txt')
common_sequences = load_common_sequences('sequences.txt')

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Password Classification")
app.geometry("750x500")

def preprocess(password):
    sequence = tokenizer.texts_to_sequences([password])
    padded_sequence = pad_sequences(sequence, maxlen=SEQUENCE_LEN)
    return padded_sequence

def check_date_sequence(password, dob):
    date_parts = dob.split('-')
    year, month, day = date_parts[0], date_parts[1], date_parts[2]
    
    sequences_to_check = [
        month + day,
        day + month,
        year + month,
        month + year,
        year + day,
        day + year,
    ]
    
    for seq in sequences_to_check:
        if seq in password:
            return True
    return False

def predict_strength(password, name, surname, dob):
    missing_criteria = []
    
    if name.lower() in password.lower() or surname.lower() in password.lower() or dob.replace('.', '').replace('-', '') in password:
        missing_criteria.append("Do not include personal information like name, surname, or date of birth.")
    
    if check_date_sequence(password, dob):
        missing_criteria.append("Do not include sequential parts of your date of birth.")
    
    if len(password) < 8:
        missing_criteria.append("Use at least 8 characters.")
    
    if not any(char.isupper() for char in password):
        missing_criteria.append("Include at least one uppercase letter.")
    
    if not any(char.isdigit() for char in password):
        missing_criteria.append("Include at least one digit.")
    
    if not any(char in "!@#$%^&*()-_+=[]{}|:;,.?" for char in password):
        missing_criteria.append("Include at least one special character.")

    for word in common_words:
        if word in password.lower():
            missing_criteria.append(f"Password contains a common word '{word}'. Please choose a stronger password.")

    for sequence in common_sequences:
        if sequence in password.lower():
            missing_criteria.append(f"Password contains a common sequence '{sequence}'. Please choose a stronger password.")
    
    if missing_criteria:
        return 0, "Weak", " ".join(missing_criteria)

    preprocessed_password = preprocess(password)
    prediction = model.predict(preprocessed_password)
    strength = np.argmax(prediction, axis=1)[0]

    if strength == 0:
        return 0, "Weak", "Password meets all basic criteria."
    elif strength == 1:
        return 1, "Medium", "Password meets all basic criteria."
    else:
        return 2, "High", "Password meets all basic criteria."

def on_button_click():
    personal_info = entry_personal_info.get()
    password = entry_password.get()

    try:
        name, surname, dob_str = personal_info.split()
        dob = datetime.strptime(dob_str, "%Y.%m.%d").strftime("%Y-%m-%d")

        _, strength_text, feedback = predict_strength(password, name, surname, dob)
        output1_text = f"Password Strength: {strength_text}"
        output2_text = feedback
    except ValueError:
        output1_text = "Entered Personal Info: Invalid format. Please enter 'Name Surname YYYY.MM.DD'."
        output2_text = "Could not parse Date of Birth, showing raw input."

    label_output1.configure(text=output1_text)
    label_output2.configure(text=output2_text)

def toggle_password():
    if show_password_var.get():
        entry_password.configure(show="")
    else:
        entry_password.configure(show="*")

def generate_password():
    length = random.randint(10, 25)
    
    password = [
        random.choice(string.ascii_lowercase),
        random.choice(string.ascii_uppercase),
        random.choice(string.digits),
        random.choice("!@#$%^&*()-_+=[]{}|:;,.?")
    ]
    
    characters = string.ascii_letters + string.digits + "!@#$%^&*()-_+=[]{}|:;,.?"
    for _ in range(length - 4):
        password.append(random.choice(characters))
    
    random.shuffle(password)
    password = ''.join(password)
    
    for word in common_words:
        if word in password.lower():
            return generate_password()
    for sequence in common_sequences:
        if sequence in password.lower():
            return generate_password()

    return password

def on_generate_password_click():
    generated_password = generate_password()
    entry_password.delete(0, 'end')
    entry_password.insert(0, generated_password)
    label_generated_password.configure(text=f"Generated Password: {generated_password}")

entry_personal_info = ctk.CTkEntry(app, placeholder_text="Name Surname YYYY.MM.DD", width=400)
entry_personal_info.pack(pady=20)

entry_password = ctk.CTkEntry(app, placeholder_text="Password", show="*", width=400)
entry_password.pack(pady=20)

show_password_var = ctk.BooleanVar()
checkbox_show_password = ctk.CTkCheckBox(app, text="Show Password", variable=show_password_var, command=toggle_password)
checkbox_show_password.pack(pady=5)

button_generate_password = ctk.CTkButton(app, text="Generate Password", command=on_generate_password_click)
button_generate_password.pack(pady=10)

label_generated_password = ctk.CTkLabel(app, text="Generated Password: ", wraplength=400)
label_generated_password.pack(pady=10)

button_submit = ctk.CTkButton(app, text="Submit", command=on_button_click)
button_submit.pack(pady=20)

label_output1 = ctk.CTkLabel(app, text="Output1: ", wraplength=400)
label_output1.pack(pady=10)
label_output2 = ctk.CTkLabel(app, text="Output2: ", wraplength=400)
label_output2.pack(pady=10)

app.mainloop()
