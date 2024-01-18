import tkinter as tk
from tkinter import filedialog

def browse_files():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                          filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: " + filename)

app = tk.Tk()
app.title("Simple GUI with File Browser")

label_file_explorer = tk.Label(app, text="File Explorer", width=100, height=4, fg="blue")
label_file_explorer.pack()

button_explore = tk.Button(app, text="Browse Files", command=browse_files)
button_explore.pack()

button2 = tk.Button(app, text="Button 2")
button2.pack()

button3 = tk.Button(app, text="Button 3")
button3.pack()

button4 = tk.Button(app, text="Button 4")
button4.pack()

app.mainloop()