import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import numpy as np


cat_pipeline = joblib.load('models\cat_model.pkl')
goal_pipeline = joblib.load('models\goal_model.pkl')
success_pipeline = joblib.load('models\success_model.pkl')

def suggest_category(text):
    return cat_pipeline.predict([text])[0]

def suggest_goal(text, category, country, duration):
    X_new = pd.DataFrame([{
        'text': text,
        'category_parent_name': category,
        'country': country,
        'campaign_duration': duration
    }])
    
    log_goal = goal_pipeline.predict(X_new)[0]
    return np.expm1(log_goal)

def predict_success(text, category, country, duration, goal):
    X_new = pd.DataFrame([{
        'text': text,
        'category_parent_name': category,
        'country': country,
        'campaign_duration': duration,
        'goal_usd': goal
    }])
    
    prob = success_pipeline.predict_proba(X_new)[0][1]
    return prob

def generate_report(title, blurb, duration, country):
    text = title + " " + blurb
    
    category = suggest_category(text)
    goal = suggest_goal(text, category, country, duration)
    prob = predict_success(text, category, country, duration, goal)
    
    return category, goal, prob


root = tk.Tk()
root.title("Asistente de Crowdfunding")
root.geometry("600x500")


tk.Label(root, text="Título del proyecto").pack()
title_entry = tk.Entry(root, width=60)
title_entry.pack()

tk.Label(root, text="Descripción").pack()
blurb_text = tk.Text(root, height=5, width=60)
blurb_text.pack()

tk.Label(root, text="Duración (días)").pack()
duration_entry = tk.Entry(root)
duration_entry.pack()

tk.Label(root, text="País").pack()

countries = ['US', 'AU', 'SE', 'ES', 'FR', 'IT', 'GB', 'PL', 'CH', 'DE', 'HK',
             'NL', 'NZ', 'MX', 'CA', 'IE', 'BE', 'JP', 'SG', 'DK', 'GR', 'AT',
             'SI', 'NO', 'LU']

country_combo = ttk.Combobox(root, values=countries)
country_combo.set("US")
country_combo.pack()


result_label = tk.Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=20)

def on_generate():
    try:
        title = title_entry.get()
        blurb = blurb_text.get("1.0", tk.END).strip()
        duration = int(duration_entry.get())
        country = country_combo.get()
        
        category, goal, prob = generate_report(title, blurb, duration, country)
        
        result_text = f"""


Categoría sugerida: {category}
Meta proyectada: ${int(goal):,}
Probabilidad de éxito: {prob:.2%}
"""
        result_label.config(text=result_text)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Generar reporte", command=on_generate).pack(pady=10)

root.mainloop()