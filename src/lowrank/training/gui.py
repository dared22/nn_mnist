import tkinter as tk
from tkinter import filedialog

app = ctk.CTk()
app.title("Enhanced GUI with CustomTkinter")

# Configure grid layout (2 columns)
app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=1)

# File Explorer label
label_file_explorer = ctk.CTkLabel(app, text="File Explorer", width=100, height=40, fg_color="gray", text_color="white")
label_file_explorer.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

# Browse Files button
button_explore = ctk.CTkButton(app, text="Browse Files", command=browse_files)
button_explore.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

# Train NN button
button_training = ctk.CTkButton(app, text="Train NN", command=start_training_thread)
button_training.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

# Predict a Number button
button_predict = ctk.CTkButton(app, text="Predict a Number from mnist", command=predict_btn)
button_predict.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

# Draw a Number button
button_draw = ctk.CTkButton(app, text="Draw a number")
button_draw.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

# Output area
output_area = st.ScrolledText(app, height=10)
output_area.grid(row=4, column=0, columnspan=2, padx=10, sticky="ew")
# Redirect stdout to the output area
sys.stdout = TextRedirector(output_area)


app.mainloop()


# Reset stdout when the app closes
sys.stdout = sys.__stdout__
