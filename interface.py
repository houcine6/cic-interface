import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import tensorflow_addons as tfa
from keras.models import load_model
from PIL import Image
import os
import threading
import time
import cv2

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, name='patch_encoder'):
        super(PatchEncoder, self).__init__(name=name)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        return cls(
            num_patches=config['num_patches'],
            projection_dim=config['projection_dim'],
            name=config['name']
        )



ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Create the main window and set its title
window = ctk.CTk()
window.title("Cic-Ids-2017")
#window.configure(bg="#444654")
window_width = 1080
window_height = 690
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")
window.resizable(False, False)

"""window.grid_rowconfigure(0, weight=1)
window.columnconfigure(1, weight=1)"""

# Define empty DataFrames
df = pd.DataFrame()
df_test = pd.DataFrame()
testdf_categorical_values = pd.DataFrame()
newdf = pd.DataFrame()
newdf_test = pd.DataFrame()
numeric_col = pd.DataFrame()
numeric_col_test = pd.DataFrame()

# create navigation frame
navigation_frame = ctk.CTkFrame(window, corner_radius=20, width=200, height=150)
navigation_frame.place(x=15, y=15)

# create navigation frame
navigation_frame2 = ctk.CTkScrollableFrame(window, corner_radius=20, width=155, height=445)
navigation_frame2.place(x = 15, y = 185)

def btn_detection():
    panel1.place(x = 220, y = 15)
    panel2.place_forget()
    panel3.place_forget()
    panel4.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel7.place_forget()
    panel8.place_forget()

button_1 = ctk.CTkButton(master=navigation_frame, width=150, height=70, corner_radius=10, text="Detection",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 16),command=btn_detection)
button_1.pack(pady=(40,40), padx=(20, 20))

def btn_Et1():
    panel2.place(x = 220, y = 15)
    panel1.place_forget()
    panel3.place_forget()
    panel4.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel7.place_forget()
    panel8.place_forget()

button_2 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Visualisation",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 16), command=btn_Et1)
button_2.pack(pady=(30,0), padx=(0, 0))

# Load the image
image = Image.open("icons/arrowDown.png")
tk_image = ImageTk.PhotoImage(image)
home_frame_large_image_label2 = ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label2.pack(pady=(15,0), padx=(20, 20))

def btn_Et2():
    global enc_loded
    global df_categorical_values
    panel3.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel4.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel7.place_forget()
    panel8.place_forget()

    if enc_loded == 0 :
        df_categorical_values = pd.read_csv("categorical_values.csv", header=None, nrows=11)
        for _, row in df_categorical_values.head(11).iterrows():
            table_before.insert("", tk.END, values=row.tolist(),tags=(mytag))
        table_before.pack()
        enc_loded =1
        del df_categorical_values


enc_loded = 0
button_3 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Encoding & Cleaning",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et2)
button_3.pack(pady=(15,0), padx=(0, 0))

# Load the image
home_frame_large_image_label3= ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label3.pack(pady=(15,0), padx=(20, 20))

def btn_Et3():
    global Normalization
    panel4.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel3.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel7.place_forget()
    panel8.place_forget()
    
    if Normalization == 0:
        all_data2 = pd.read_csv("train_data_cleande.csv", header=None, nrows=11)
        # Add each row of the selected columns to the table without the index
        for _, row in all_data2.head(11).iterrows():
            table_normal.insert("", tk.END, values=row.tolist(),tags=(mytag))
        table_normal.place(x=0, y=0, width=950)
        del all_data2
        Normalization = 1

Normalization = 0
button_4 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Normalization",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et3)
button_4.pack(pady=(15,0), padx=(0, 0))

# Load the image
home_frame_large_image_label4 = ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label4.pack(pady=(15,0), padx=(20, 20))

def btn_Et4():
    panel5.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel4.place_forget()
    panel3.place_forget()
    panel6.place_forget()
    panel7.place_forget()
    panel8.place_forget()

button_5 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Feature Selection",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et4)
button_5.pack(pady=(15,0), padx=(0, 0))

# Load the image
home_frame_large_image_label5 = ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label5.pack(pady=(15,0), padx=(20, 20))

def btn_Et5():
    panel6.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel4.place_forget()
    panel3.place_forget()
    panel5.place_forget()
    panel7.place_forget()
    panel8.place_forget()

button_6 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Label Encoding",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et5)
button_6.pack(pady=(15,00), padx=(0, 0))

# Load the image
home_frame_large_image_label6= ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label6.pack(pady=(15,0), padx=(20, 20))

def btn_Et6():
    panel7.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel4.place_forget()
    panel3.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel8.place_forget()

button_7 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Data Sampling",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et6)
button_7.pack(pady=(15,00), padx=(0, 0))

# Load the image
home_frame_large_image_label7= ctk.CTkLabel(master=navigation_frame2, text="", image=tk_image)
home_frame_large_image_label7.pack(pady=(15,0), padx=(20, 20))

def btn_Et7():
    panel8.place(x = 220, y = 15)
    panel1.place_forget()
    panel2.place_forget()
    panel4.place_forget()
    panel3.place_forget()
    panel5.place_forget()
    panel6.place_forget()
    panel7.place_forget()

    if loded == 0 :
        n = pd.read_csv("attacks_balanced_4/0.csv", header=None, nrows=30)
        for _, row in n.head(30).iterrows():
            tablen.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tablen.place(x=0, y=0, width=400)

        d = pd.read_csv("attacks_balanced_4/1.csv", header=None, nrows=30)
        for _, row in d.head(30).iterrows():
            tabled.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tabled.place(x=0, y=0, width=400)

        dd = pd.read_csv("attacks_balanced_4/2.csv", header=None, nrows=30)
        for _, row in dd.head(30).iterrows():
            tabledd.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tabledd.place(x=0, y=0, width=400)

        w = pd.read_csv("attacks_balanced_4/3.csv", header=None, nrows=30)
        for _, row in w.head(30).iterrows():
            tablew.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tablew.place(x=0, y=0, width=400)

        b = pd.read_csv("attacks_balanced_4/4.csv", header=None, nrows=30)
        for _, row in b.head(30).iterrows():
            tableb.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tableb.place(x=0, y=0, width=400)

        s = pd.read_csv("attacks_balanced_4/5.csv", header=None, nrows=30)
        for _, row in s.head(30).iterrows():
            tables.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tables.place(x=0, y=0, width=400)

        h = pd.read_csv("attacks_balanced_4/6.csv", header=None, nrows=30)
        for _, row in h.head(30).iterrows():
            tableh.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tableh.place(x=0, y=0, width=400)

        p = pd.read_csv("attacks_balanced_4/7.csv", header=None, nrows=30)
        for _, row in p.head(30).iterrows():
            tablep.insert("", tk.END, values=row.tolist(),tags=(mytag))
        tablep.place(x=0, y=0, width=400)

        



loded = 0
button_8 = ctk.CTkButton(master=navigation_frame2, width=150, height=50, corner_radius=10, text="Transformation",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=btn_Et7)
button_8.pack(pady=(15,30), padx=(0, 0))




#create navigation frame VISUALISATION
panel2 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel2.place(x = 220, y = 15)
panel2.pack_propagate(False)
panel2.place_forget()

def validate_entry(text):
    # Check if the entered text is a valid number
    try:
        float(text)
        return True
    except ValueError:
        return False

def load_dataset():
    global clicked
    global df
    if clicked == 0 :
        entry_1_value = entry_1.get()
        if entry_1_value == "":
            entry_1.configure(placeholder_text="Enter a percentage:", placeholder_text_color="red", border_color="red")
            print("entry_1 is empty.")
        else:
            try:
                def analysis_thread():
                    percentage = int(entry_1_value)
                    if percentage >= 1 and percentage <= 100:
                        print("entry_1_value is a valid integer between 1 and 100.")
                        # Specify the number of rows you want to read
                        taille = int(entry_1_value) * 2183541 // 100  # Use integer division (//) to ensure an integer result
                        df = pd.read_csv("train.csv", header=None, nrows=taille)
                        # Add each row of the selected columns to the table without the index
                        for _, row in df.head(11).iterrows():
                            table.insert("", tk.END, values=row.tolist(),tags=(mytag))
                        label_loading_t.place_forget()
                        table_width = 800  # set the desired width of the table
                        table.place(x=0, y=0, width=table_width)
                        label_shape_train.configure(text="("+ str(taille) + ", " + str(df.shape[1]) + ")")
                        label_shape_train.place(relx=0.09, rely=0.31, anchor='center')
                        button_load_dataset.configure(text="Clear Dataset")
                        entry_1.configure(placeholder_text_color="gray", border_color="gray", text_color="gray")
                        button_load_dataset.focus_set()
                    else:
                        print("entry_1_value is not within the valid range of 1 to 100.")
                        entry_1.configure(placeholder_text="Enter a percentage:", text_color="red", border_color="red")
                clicked = 1
                # Create and start the analysis thread
                thread = threading.Thread(target=analysis_thread)
                thread.start()
                label_loading_t.place(relx=0.50, rely=0.50, anchor='center')
            except ValueError:
                print("entry_1_value is not an integer.")
                entry_1.configure(placeholder_text="Enter a percentage:", text_color="red", border_color="red")
    
    elif clicked == 1 :
        table.delete(*table.get_children())
        table.place_forget()
        label_shape_train.place_forget()
        df = None  # Set df to None when clearing the dataset
        del df
        entry_1.delete(0, 'end')  # Clear the text in the entry widget
        entry_1.configure(placeholder_text="Enter a percentage:", )
        button_load_dataset.focus_set()
        button_load_dataset.configure(text="Load \nTrain Dataset")
        clicked = 0

clicked = 0
button_load_dataset = ctk.CTkButton(master=panel2, width=0, height=60, corner_radius=50, text="Load \nTrain Dataset",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 12), command=load_dataset,
                          border_width=1, border_color="#4c4d52")
button_load_dataset.place(relx=0.01, rely=0.10)
entry_1 = ctk.CTkEntry(master=panel2, placeholder_text="Enter a percentage:")
entry_1.place(relx=0.01, rely=0.22)

# Create the tab view
tab_view = ctk.CTkTabview(panel2, corner_radius=15, fg_color="gray11", width=677, height=250)
tab_view.place(relx=0.59, rely=0.22, anchor="center")
# Create a tab and add a listbox to it
tab = tab_view.add("CIC-IDS-2017 Train:")
table = ttk.Treeview(tab, columns=('Flow ID', ' Source IP', ' Source Port', ' Destination IP',
            ' Destination Port', ' Protocol', ' Timestamp', ' Flow Duration',
            ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd Packet Length Max', ' Fwd Packet Length Min',
            ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
            'Bwd Packet Length Max', ' Bwd Packet Length Min',
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
            ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
            ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
            ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
            ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
            ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
            ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
            ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
            'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
            ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
            ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
            ' Average Packet Size', ' Avg Fwd Segment Size',
            ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
            ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
            ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
            ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
            'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
            ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
            ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
            ' Idle Max', ' Idle Min', ' Label'), show = '', height=11)
table.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table["columns"]:
    table.column(column, width=100)
# Create a horizontal scrollbar
xscrollbar = ctk.CTkScrollbar(tab_view, orientation='horizontal', command=table.xview, width=650)
xscrollbar.place(x=16, y=230)
# Set the xscrollbar to control the table's x-axis
table.configure(xscrollcommand=xscrollbar.set)
table.pack_forget()

label_loading_t = ctk.CTkLabel(master=tab, justify=ctk.LEFT, text="Loading...", font=("Roboto", 17))
label_loading_t.place_forget()

label_shape_train = ctk.CTkLabel(master=panel2, justify=ctk.LEFT, text="", font=("Roboto", 17))
label_shape_train.place_forget()

label_shape_test = ctk.CTkLabel(master=panel2, justify=ctk.LEFT, text="", font=("Roboto", 17))
label_shape_test.place_forget()

def load_dataset_tst():
    global clicked_tst
    global df_test
    if clicked_tst == 0 :
        entry_2_value = entry_2.get()
        if entry_2_value == "":
            entry_2.configure(placeholder_text="Enter a percentage:", placeholder_text_color="red", border_color="red")
            print("entry_2 is empty.")
        else:
            try:

                def analysis_thread():
                    percentage = int(entry_2_value)
                    if percentage >= 1 and percentage <= 100:
                        print("entry_2_value is a valid integer between 1 and 100.")
                        # Specify the number of rows you want to read
                        taille = int(entry_2_value) * 935804 // 100  # Use integer division (//) to ensure an integer result
                        df_test = pd.read_csv("test.csv", header=None, nrows=taille)
                        # Add each row of the selected columns to the table without the index
                        for _, row in df_test.head(11).iterrows():
                            table_tst.insert("", tk.END, values=row.tolist(),tags=(mytag))
                        label_loading_tst.place_forget()
                        table_width = 800  # set the desired width of the table
                        table_tst.place(x=0, y=0, width=table_width)
                        label_shape_test.configure(text="("+ str(taille) + ", " + str(df_test.shape[1]) + ")")
                        label_shape_test.place(relx=0.09, rely=0.75, anchor='center')
                        entry_2.configure(placeholder_text_color="gray", border_color="gray", text_color="gray")
                        button_load_dataset_tst.configure(text="Clear Dataset")
                        button_load_dataset_tst.focus_set()
                        clicked_tst = 1
                    else:
                        print("entry_2_value is not within the valid range of 1 to 100.")
                        entry_2.configure(placeholder_text="Enter a percentage:", text_color="red", border_color="red")
                
                clicked_tst = 1
                # Create and start the analysis thread
                thread = threading.Thread(target=analysis_thread)
                thread.start()
                label_loading_tst.place(relx=0.50, rely=0.50, anchor='center')

            except ValueError:
                print("entry_2_value is not an integer.")
                entry_2.configure(placeholder_text="Enter a percentage:", text_color="red", border_color="red")
    elif clicked_tst == 1 :
        table_tst.delete(*table_tst.get_children())
        table_tst.place_forget()
        label_shape_test.place_forget()
        entry_2.delete(0, 'end')  # Clear the text in the entry widget
        entry_2.configure(placeholder_text="Enter a percentage:", )
        button_load_dataset_tst.focus_set()
        button_load_dataset_tst.configure(text="Load \nTrain Dataset")
        df_test = None  # Set df to None when clearing the dataset
        del df_test
        clicked_tst = 0

clicked_tst = 0
button_load_dataset_tst = ctk.CTkButton(master=panel2, width=0, height=60, corner_radius=50, text="Load \nTest Dataset",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 12), command=load_dataset_tst,
                          border_width=1, border_color="#4c4d52")
button_load_dataset_tst.place(relx=0.01, rely=0.53)
entry_2 = ctk.CTkEntry(master=panel2, placeholder_text="Enter a percentage:")
entry_2.place(relx=0.01, rely=0.66)
# Create the tab view
tab_view_tst = ctk.CTkTabview(panel2, corner_radius=15, fg_color="gray11", width=677, height=250)
tab_view_tst.place(relx=0.59, rely=0.65, anchor="center")
# Create a tab and add a listbox to it
tab_tst = tab_view_tst.add("CIC-IDS-2017 Test:")
table_tst = ttk.Treeview(tab_tst, columns=('Flow ID', ' Source IP', ' Source Port', ' Destination IP',
                    ' Destination Port', ' Protocol', ' Timestamp', ' Flow Duration',
                    ' Total Fwd Packets', ' Total Backward Packets',
                    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                    ' Fwd Packet Length Max', ' Fwd Packet Length Min',
                    ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                    'Bwd Packet Length Max', ' Bwd Packet Length Min',
                    ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
                    ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
                    ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                    ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
                    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
                    ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
                    ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
                    ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
                    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
                    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
                    ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
                    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
                    ' Average Packet Size', ' Avg Fwd Segment Size',
                    ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
                    ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                    ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                    ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                    'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
                    ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
                    ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
                    ' Idle Max', ' Idle Min', ' Label'), show = '', height=11)
table_tst.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag_tst = 'gray'
# Set the width of all columns to 50 pixels
for column in table_tst["columns"]:
    table_tst.column(column, width=100)
# Create a horizontal scrollbar
xscrollbar_tst = ctk.CTkScrollbar(tab_view_tst, orientation='horizontal', command=table_tst.xview, width=650)
xscrollbar_tst.place(x=16, y=230)
# Set the xscrollbar to control the table's x-axis
table_tst.configure(xscrollcommand=xscrollbar_tst.set)
table_tst.pack_forget()

label_loading_tst = ctk.CTkLabel(master=tab_tst, justify=ctk.LEFT, text="Loading...", font=("Roboto", 17))
label_loading_tst.place_forget()

def load_chart():
    test_data = pd.read_csv('test.csv', usecols=[' Label'])
    # Calculate value counts of 'Label' column
    label_counts = test_data[' Label'].value_counts()
    # Create a bar chart
    label_counts.plot(kind='bar')
    # Set labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Classes')
    # Display the chart
    plt.show()
    del train_data

# Load the image
barimg = Image.open("icons/bar.png")
barimg = ImageTk.PhotoImage(barimg)
button_Show = ctk.CTkButton(master=panel2, width=0, height=60, corner_radius=50, text="Display Chart",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 15), command=load_chart,
                          border_width=4, border_color="#4c4d52", image=barimg)
button_Show.place(relx=0.5, rely=0.93, anchor='center')


#create navigation frame
panel3 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel3.place(x = 220, y = 15)
panel3.place_forget()

# Create the tab view
tab_view_codage = ctk.CTkTabview(panel3, corner_radius=15, fg_color="gray11", width=250, height=250)
tab_view_codage.place(relx=0.02, rely=0.02)
# Create a tab and add a listbox to it
tab_before = tab_view_codage.add("Before Encoding:")
table_before = ttk.Treeview(tab_before, columns=('Flow ID', ' Source IP', ' Destination IP', ' Timestamp'), show = '', height=11)
table_before.tag_configure('gray',background="gray20", foreground="white", font=("Roboto",8))
mytag_tst = 'gray'
# Set the width of all columns to 50 pixels
for column in table_before["columns"]:
    table_before.column(column, width=80)
#table_before.pack()
table_before.pack_forget()

def encode():
    global encoded
    if encoded == 0 :

        def analysis_thread():
            all_data_categorical_values = pd.read_csv('categorical_values.csv')
            all_data_categorical_values = all_data_categorical_values.apply(LabelEncoder().fit_transform)

            all_data_categorical_values = all_data_categorical_values.add_suffix('_enc')

            # add data to the table
            table_after.insert("", tk.END, values=('Flow ID', ' Source IP', ' Destination IP', ' Timestamp'),tags=mytag)

            # Add each row of the selected columns to the table without the index
            for _, row in all_data_categorical_values.head(10).iterrows():
                table_after.insert("", tk.END, values=row.tolist(),tags=(mytag))
            label_loading_enc.place_forget()
            table_after.pack()
            del all_data_categorical_values
            # encode_pt2()
        
        encoded = 1
        # Create and start the analysis thread
        thread = threading.Thread(target=analysis_thread)
        thread.start()
        label_loading_enc.place(relx=0.50, rely=0.50, anchor='center')



encoded = 0
# Load the image
imgarrow = Image.open("icons/double_arrow.png")
imgarrow = ImageTk.PhotoImage(imgarrow)
button_encode = ctk.CTkButton(master=panel3, width=0, height=60, corner_radius=50, text="Encode",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 15), command=encode,
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right")
button_encode.place(relx=0.50, rely=0.22, anchor='center')

# Create the tab view
tab_view_codage_af = ctk.CTkTabview(panel3, corner_radius=15, fg_color="gray11", width=250, height=250)
tab_view_codage_af.place(relx=0.625, rely=0.02)
# Create a tab and add a listbox to it
tab_after = tab_view_codage_af.add("After Encoding:")
table_after = ttk.Treeview(tab_after, columns=('Flow ID', ' Source IP', ' Destination IP', ' Timestamp'), show = '', height=11)
table_after.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag_af = 'gray'
# Set the width of all columns to 50 pixels
for column in table_after["columns"]:
    table_after.column(column, width=80)
#table_before.pack()
table_after.pack_forget()

label_loading_enc = ctk.CTkLabel(master=tab_view_codage_af, justify=ctk.LEFT, text="Loading...", font=("Roboto", 17))
label_loading_enc.place_forget()

# Create the tab view
tab_view_nettoyage = ctk.CTkTabview(panel3, corner_radius=15, fg_color="gray11", width=800, height=300)
tab_view_nettoyage.place(relx=0.02, rely=0.5)
# Create a tab and add a listbox to it
tab_nettoyage = tab_view_nettoyage.add("Drop NaN values")

def drop_Nan():
    global normalised
    if normalised == 0 :
        label_before_nan.configure(text="Before: (2183541,85)")
        train_data2_dropped_rows = pd.read_csv("train_data_cleande.csv",nrows=10)
        label_After_nan.configure(text="   After: (1979339, 77)")
        label_before_nan.place(relx=0.11, rely=0.60, anchor='center')
        label_After_nan.place(relx=0.1, rely=0.70, anchor='center')

        new_table.insert("", tk.END, values=(' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
                                    ' Total Fwd Packets', ' Total Backward Packets',
                                    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
                                    ' Fwd Packet Length Max', ' Fwd Packet Length Min',
                                    ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                                    'Bwd Packet Length Max', ' Bwd Packet Length Min',
                                    ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
                                    ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
                                    ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                                    ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
                                    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
                                    ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
                                    'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
                                    ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
                                    ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
                                    ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
                                    ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
                                    ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
                                    ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
                                    ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                                    'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
                                    ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
                                    ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
                                    ' Idle Max', ' Idle Min', ' Label', 'Flow ID_enc', 'Source IP_enc',
                                    'Destination IP_enc', 'Timestamp_enc'),tags=(mytag))
        
        # Add each row of the selected columns to the table without the index
        for _, row in train_data2_dropped_rows.head(11).iterrows():
            new_table.insert("", tk.END, values=row.tolist(),tags=(mytag))
        table_width = 750  # set the desired width of the table
        new_table.place(x=230, y=80, width=table_width)
        normalised = 1

normalised = 0
button_clean = ctk.CTkButton(master=tab_view_nettoyage, width=0, height=60, corner_radius=50, text="Clean Dataset",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=drop_Nan,
                          border_width=4, border_color="#4c4d52")
button_clean.place(relx=0.115, rely=0.35, anchor='center')

label_before_nan = ctk.CTkLabel(master=tab_view_nettoyage, justify=ctk.LEFT, text="", font=("Roboto", 17))
label_before_nan.place(relx=0.1, rely=0.50, anchor='center')
label_before_nan.place_forget()

label_After_nan = ctk.CTkLabel(master=tab_view_nettoyage, justify=ctk.LEFT, text="", font=("Roboto", 17))
label_After_nan.place(relx=0.1, rely=0.60, anchor='center')
label_After_nan.place_forget()

new_table = ttk.Treeview(tab_view_nettoyage, columns=(' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
       ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
       'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
       ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
       ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
       ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
       ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
       ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min', ' Label', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc'), show = '', height=11)

new_table.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in new_table["columns"]:
    new_table.column(column, width=100)
# Create a horizontal scrollbar
xscrollbar_newtable = ctk.CTkScrollbar(tab_view_nettoyage, orientation='horizontal', command=new_table.xview, width=605)
xscrollbar_newtable.place(x=180, y=240)
# Set the xscrollbar to control the table's x-axis
new_table.configure(xscrollcommand=xscrollbar_newtable.set)
new_table.pack_forget()

#create navigation frame
panel4 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel4.place(x = 220, y = 15)
panel4.place_forget()

def load_normal_dataset():
    global normal
    if normal == 0:
        NOR = pd.read_csv("normalise.csv", header=None, nrows=11)
        # Drop the first column
        NOR = NOR.iloc[:, 1:]
        # Add each row of the selected columns to the table without the index
        for _, row in NOR.head(11).iterrows():
            values = [round(val, 6) if isinstance(val, float) else val for val in row.tolist()]
            table_normal2.insert("", tk.END, values=values, tags=(mytag))
            #table_normal2.insert("", tk.END, values=row.tolist(),tags=(mytag))
        table_normal2.place(x=0, y=0, width=950)
        del NOR
        normal = 1

    """if newdf is not None and not newdf.empty and newdf_test is not None and not newdf_test.empty:
        numeric_col = newdf.select_dtypes(include='number').columns
        numeric_col_test = newdf_test.select_dtypes(include='number').columns

        scaler = MinMaxScaler()
        newdf[numeric_col] = scaler.fit_transform(newdf[numeric_col])
        newdf_test[numeric_col_test] = scaler.fit_transform(newdf_test[numeric_col_test])
        
        # add data to the table
        table_normal.insert("", tk.END, values=(' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
       ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
       'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
       ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
       ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
       ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
       ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
       ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min', ' Label', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc'),
                                    tags=mytag)
        
        # Add each row of the selected columns to the table without the index
        for _, row in newdf_test.head(10).iterrows():
            # Round the floating-point values to 6 digits before inserting them into the table
            values = [round(val, 6) if isinstance(val, float) else val for val in row.tolist()]
            table_normal.insert("", tk.END, values=values, tags=(mytag))
        table_width = 800  # set the desired width of the table
        table_normal.place(x=0, y=0, width=table_width)
    else:
        print("newdf or newdf_test is empty")  
        print(newdf_test.shape)
        print(newdf.shape)"""

normal = 0
button_normalize_dataset = ctk.CTkButton(master=panel4, width=0, height=60, corner_radius=50, text="Normalize ",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14), command=load_normal_dataset,
                          border_width=4, border_color="#4c4d52", image=tk_image, compound="right")
button_normalize_dataset.place(relx=0.495, rely=0.49, anchor="center")

# Create the tab view
tab_view_normal = ctk.CTkTabview(panel4, corner_radius=15, fg_color="gray11", width=800, height=250)
tab_view_normal.place(relx=0.50, rely=0.20, anchor="center")
# Create a tab and add a listbox to it
tab_nor = tab_view_normal.add("Before Normalization:")
table_normal = ttk.Treeview(tab_nor, columns=(' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
       ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
       'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
       ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
       ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
       ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
       ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
       ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min', ' Label', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc'), show = '', height=11)

table_normal.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table_normal["columns"]:
    table_normal.column(column, width=100)
# Create a horizontal scrollbar
xscrollbar_nor = ctk.CTkScrollbar(tab_view_normal, orientation='horizontal', command=table_normal.xview, width=765)
xscrollbar_nor.place(x=16, y=230)
# Set the xscrollbar to control the table's x-axis
table_normal.configure(xscrollcommand=xscrollbar_nor.set)
table_normal.pack_forget()

# Create the tab view
tab_view_normal2 = ctk.CTkTabview(panel4, corner_radius=15, fg_color="gray11", width=800, height=250)
tab_view_normal2.place(relx=0.50, rely=0.76, anchor="center")
# Create a tab and add a listbox to it
tab_nor2 = tab_view_normal2.add("After Normalization:")
table_normal2 = ttk.Treeview(tab_nor2, columns=(' Source Port', ' Destination Port', ' Protocol', ' Flow Duration',
       ' Total Fwd Packets', ' Total Backward Packets',
       'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
       ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
       ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
       ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
       ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
       ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
       ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
       'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length',
       ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
       ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
       ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
       ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
       ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
       ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean',
       ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std',
       ' Idle Max', ' Idle Min', ' Label', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc'), show = '', height=11)
table_normal2.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table_normal2["columns"]:
    table_normal2.column(column, width=130)
# Create a horizontal scrollbar
xscrollbar_nor2 = ctk.CTkScrollbar(tab_view_normal2, orientation='horizontal', command=table_normal2.xview, width=765)
xscrollbar_nor2.place(x=16, y=230)
# Set the xscrollbar to control the table's x-axis
table_normal2.configure(xscrollcommand=xscrollbar_nor2.set)
table_normal2.pack_forget()

#create navigation frame
panel5 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel5.place(x = 220, y = 15)
panel5.place_forget()

label_Var = ctk.CTkLabel(master=panel5, justify=ctk.LEFT, text="Varience", font=("Roboto", 17))
label_Var.place(relx=0.26, rely=0.015)
label_RF = ctk.CTkLabel(master=panel5, justify=ctk.LEFT, text="Random Forest Classifier", font=("Roboto", 17))
label_RF.place(relx=0.60, rely=0.015)
"""label_DT = ctk.CTkLabel(master=panel5, justify=ctk.LEFT, text="Decision Tree", font=("Roboto", 17))
label_DT.place(relx=0.77, rely=0.015)"""

# Create the tab view
tab_view_varience = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray11", width=250, height=480)
tab_view_varience.place(relx=0.15, rely=0.05)
tab_var = tab_view_varience.add("Deleted:")
tab_var2 = tab_view_varience.add("Selected:")
# function to be called when mouse enters in a frame
def pressed2(event):
    # Insert each item from the listVarience list into the listbox
    for item in listVarience:
        listbox.insert(tk.END, item)
        listbox.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listbox.pack(fill=tk.BOTH, expand=True)
    for item in listVarience2:
        listbox2.insert(tk.END, item)
        listbox2.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listbox2.pack(fill=tk.BOTH, expand=True)
    label_shape_Var.configure(text="Shape: (1979339, 38)")
    label_shape_Var.place(relx=0.50, rely=0.55, anchor='center')
    #(1979339, 38)

tab_var.bind('<Double 1>', pressed2)
# Create a listbox widget
listbox = tk.Listbox(tab_var)
listbox.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listbox.place_forget()
listbox2 = tk.Listbox(tab_var2)
listbox2.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listbox2.place_forget()
# Example usage
listVarience = ['Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'CWE Flag Count',
                'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
                'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Min', 'Flow Bytes/s',
                'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Min',
                'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Min', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Packet Length Variance', 'Fwd URG Flags', 'RST Flag Count',
                'Down/Up Ratio', 'Avg Fwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'act_data_pkt_fwd', 'Active Mean', 'Active Std',
                'Active Max', 'Active Min', 'Idle Std', 'ECE Flag Count','Average Packet Size','Max Packet Length']
listVarience2 = [' Source Port', ' Destination Port', ' Protocol',
       ' Flow Duration', 'Bwd Packet Length Max', ' Bwd Packet Length Mean',
       ' Bwd Packet Length Std', ' Flow IAT Max', 'Fwd IAT Total',
       ' Fwd IAT Std', ' Fwd IAT Max', 'Bwd IAT Total', ' Bwd IAT Max',
       'Fwd PSH Flags', ' Fwd Header Length', ' Bwd Header Length',
       ' Packet Length Mean', ' Packet Length Std', 'FIN Flag Count',
       ' SYN Flag Count', ' PSH Flag Count', ' ACK Flag Count',
       ' URG Flag Count', ' Avg Bwd Segment Size', ' Fwd Header Length.1',
       'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
       ' min_seg_size_forward', 'Idle Mean', 'Idle Std', ' Idle Max', ' Idle Min',
       ' Label', 'Flow ID_enc', 'Source IP_enc', 'Destination IP_enc',
       'Timestamp_enc']
tab_view_varience_R = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray30", width=250, height=110)
tab_view_varience_R.place(relx=0.15, rely=0.79)
label_shape_Var = ctk.CTkLabel(master=tab_view_varience_R, justify=ctk.LEFT, text="", font=("Roboto", 20))
label_shape_Var.place_forget()


# Create the tab view
tab_view_RF = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray11", width=250, height=480)
tab_view_RF.place(relx=0.71, rely=0.415, anchor="center")
tab_RF = tab_view_RF.add("Deleted:")
tab_RF2 = tab_view_RF.add("Selected:")
# function to be called when mouse enters in a frame
def pressed1(event):
    # Insert each item from the listVarience list into the listbox
    for item in RF:
        listboxRF.insert(tk.END, item)
        listboxRF.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listboxRF.pack(fill=tk.BOTH, expand=True)
    for item in RF_selected:
        listboxRF2.insert(tk.END, item)
        listboxRF2.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listboxRF2.pack(fill=tk.BOTH, expand=True)
    label_shape_RF.configure(text="Shape: (1979339, 31)")
    label_shape_RF.place(relx=0.50, rely=0.55, anchor='center')

tab_RF.bind('<Double 1>', pressed1)
# Create a listbox widget
listboxRF = tk.Listbox(tab_RF)
listboxRF.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listboxRF.place_forget()
listboxRF2 = tk.Listbox(tab_RF2)
listboxRF2.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listboxRF2.place_forget()
# Example usage
RF = [
    'Flow Bytes/s',
    'act_data_pkt_fwd',
    'Fwd IAT Max',
    'Fwd Packets/s',
    'Flow Duration',
    'Subflow Fwd Packets',
    'Subflow Bwd Packets',
    'Total Backward Packets',
    'Fwd IAT Total',
    'Flow IAT Std',
    'Idle Max',
    'Fwd IAT Mean',
    'Idle Mean',
    'ACK Flag Count',
    'Subflow Bwd Bytes',
    'Flow Packets/s',
    'Min Packet Length',
    'Idle Min',
    'Down/Up Ratio',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Min',
    'Fwd IAT Min',
    'Bwd IAT Mean',
    'Bwd IAT Total',
    'Bwd IAT Std',
    'Active Mean',
    'Flow IAT Min',
    'Bwd IAT Max',
    'Active Min',
    'FIN Flag Count',
    'Active Max',
    'URG Flag Count',
    'SYN Flag Count',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Idle Std',
    'Active Std',
    'Bwd Header Length',
    'Protocol',
    'Fwd Header Length.1',
    'Fwd Header Length',
    'CWE Flag Count',
    'Fwd URG Flags',
    'RST Flag Count',
    'min_seg_size_forward',
    'ECE Flag Count'
]
RF_selected = [' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', ' Label', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc']
tab_view_RF_R = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray30", width=250, height=110)
tab_view_RF_R.place(relx=0.56, rely=0.79)
label_shape_RF = ctk.CTkLabel(master=tab_view_RF_R, justify=ctk.LEFT, text="", font=("Roboto", 20))
label_shape_RF.place_forget()

"""
# Create the tab view
tab_view_DT = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray11", width=250, height=480)
tab_view_DT.place(relx=0.685, rely=0.05)
tab_DT = tab_view_DT.add("Deleted:")
tab_DT2 = tab_view_DT.add("Selected:")
# function to be called when mouse enters in a frame
def pressed3(event):
    # Insert each item from the listVarience list into the listbox
    for item in DT:
        listboxDT.insert(tk.END, item)
        listboxDT.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listboxDT.pack(fill=tk.BOTH, expand=True)
    for item in DT2:
        listboxDT2.insert(tk.END, item)
        listboxDT2.insert(tk.END, "------- ------- -------")  # Insert an empty string as a line
    listboxDT2.pack(fill=tk.BOTH, expand=True)
    label_shape_DT.configure(text="Shape: (1979339, 25)")
    label_shape_DT.place(relx=0.50, rely=0.55, anchor='center')
    #(1979339, 38)

tab_DT.bind('<Double 1>', pressed3)
# Create a listbox widget
listboxDT = tk.Listbox(tab_DT)
listboxDT.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listboxDT.place_forget()
listboxDT2 = tk.Listbox(tab_DT2)
listboxDT2.configure(background='gray11', borderwidth=0, font=("Roboto", 15), foreground="gray75", justify="center",
                   activestyle="dotbox", selectbackground='gray')
listboxDT2.place_forget()
# Example usage
DT = [ ' Protocol',
          ' Total Fwd Packets', ' Total Backward Packets',
          'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
          ' Fwd Packet Length Max', ' Fwd Packet Length Min',
        ' Fwd Packet Length Std',
          'Bwd Packet Length Max',
        ' Bwd Packet Length Std', 'Flow Bytes/s',
          ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max',
          ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean',
           'Bwd IAT Total', ' Bwd IAT Mean',
          ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags',
          ' Fwd URG Flags', ' Bwd Header Length',
          ' Bwd Packets/s', ' Min Packet Length',
          ' Max Packet Length', ' Packet Length Std',
          ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count',
          ' RST Flag Count', ' ACK Flag Count',
          ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
          ' Down/Up Ratio',
          ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets',
            ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
          'Init_Win_bytes_forward',
          ' Active Std', ' Active Min', 'Idle Mean', ' Idle Std',
           'Flow ID_enc',
          'Destination IP_enc', 'Timestamp_enc']
DT2 = [' Source Port', ' Destination Port', ' Flow Duration',
       ' Fwd Packet Length Mean', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Fwd IAT Std', ' Fwd IAT Max',
       ' Fwd IAT Min', ' Fwd Header Length', 'Fwd Packets/s',
       ' Packet Length Mean', ' PSH Flag Count', ' Average Packet Size',
       ' Avg Fwd Segment Size', ' Subflow Fwd Bytes',
       ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
       ' min_seg_size_forward', 'Active Mean', ' Active Max', ' Idle Max',
       ' Idle Min', ' Label', 'Source IP_enc']
tab_view_DT_R = ctk.CTkTabview(panel5, corner_radius=15, fg_color="gray30", width=250, height=110)
tab_view_DT_R.place(relx=0.685, rely=0.79)
label_shape_DT = ctk.CTkLabel(master=tab_view_DT_R, justify=ctk.LEFT, text="", font=("Roboto", 20))
label_shape_DT.place_forget()"""





#create navigation frame
panel6 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel6.place(x = 220, y = 15)
panel6.place_forget()

tab_view_label_normal = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_normal.place(relx=0.02, rely=0.00)
label_Label_n = ctk.CTkLabel(master=tab_view_label_normal, justify=ctk.LEFT, text="Benign", font=("Roboto",14))
label_Label_n.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_dos = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_dos.place(relx=0.02, rely=0.125)
label_Label_d = ctk.CTkLabel(master=tab_view_label_dos, justify=ctk.LEFT, text="Slowloris, Slowhttptest, \n     Hulk, GoldenEye.", font=("Roboto", 13))
label_Label_d.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_ddos = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_ddos.place(relx=0.02, rely=0.245)
label_Label_dd = ctk.CTkLabel(master=tab_view_label_ddos, justify=ctk.LEFT, text="DDoS", font=("Roboto", 14))
label_Label_dd.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_Web = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_Web.place(relx=0.02, rely=0.365)
label_Label_w = ctk.CTkLabel(master=tab_view_label_Web, justify=ctk.LEFT, text="Brute Force, XSS,\n     Sql Injection", font=("Roboto", 14))
label_Label_w.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_botnet = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_botnet.place(relx=0.02, rely=0.49)
label_Label_b = ctk.CTkLabel(master=tab_view_label_botnet, justify=ctk.LEFT, text="Bot", font=("Roboto", 14))
label_Label_b.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_scan = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_scan.place(relx=0.02, rely=0.615)
label_Label_s = ctk.CTkLabel(master=tab_view_label_scan, justify=ctk.LEFT, text="PortScan", font=("Roboto", 14))
label_Label_s.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_heart = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_heart.place(relx=0.02, rely=0.74)
label_Label_h = ctk.CTkLabel(master=tab_view_label_heart, justify=ctk.LEFT, text="Heartbleed", font=("Roboto", 14))
label_Label_h.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_patator = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_patator.place(relx=0.02, rely=0.86)
label_Label_p = ctk.CTkLabel(master=tab_view_label_patator, justify=ctk.LEFT, text="SSH-Patato,\nFTP-Patator", font=("Roboto", 14))
label_Label_p.place(relx=0.50, rely=0.60, anchor ="center")

def display1():
    tab_view_label_normal.place(relx=0.40, rely=0.00)
    tab_view_label_dos.place(relx=0.40, rely=0.125)
    tab_view_label_ddos.place(relx=0.40, rely=0.245)
    tab_view_label_Web.place(relx=0.40, rely=0.365)
    tab_view_label_botnet.place(relx=0.40, rely=0.49)
    tab_view_label_scan.place(relx=0.40, rely=0.615)
    tab_view_label_heart.place(relx=0.40, rely=0.74)
    tab_view_label_patator.place(relx=0.40, rely=0.86)
    button_map_label2.place(relx=0.68, rely=0.50, anchor="center")

button_map_label = ctk.CTkButton(master=panel6, width=0, height=80, corner_radius=50, text="",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 14),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=display1)
button_map_label.place(relx=0.30, rely=0.50, anchor="center")

tab_view_label_normal = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_normal.place(relx=0.40, rely=0.00)
tab_view_label_normal.place_forget()
label_Label_n = ctk.CTkLabel(master=tab_view_label_normal, justify=ctk.LEFT, text="Normal", font=("Roboto",14))
label_Label_n.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_dos = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_dos.place(relx=0.40, rely=0.125)
tab_view_label_dos.place_forget()
label_Label_d = ctk.CTkLabel(master=tab_view_label_dos, justify=ctk.LEFT, text="DOS", font=("Roboto", 14))
label_Label_d.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_ddos = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_ddos.place(relx=0.40, rely=0.245)
tab_view_label_ddos.place_forget()
label_Label_dd = ctk.CTkLabel(master=tab_view_label_ddos, justify=ctk.LEFT, text="DDoS", font=("Roboto", 14))
label_Label_dd.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_Web = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_Web.place(relx=0.40, rely=0.365)
tab_view_label_Web.place_forget()
label_Label_w = ctk.CTkLabel(master=tab_view_label_Web, justify=ctk.LEFT, text="Web Attack", font=("Roboto", 14))
label_Label_w.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_botnet = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_botnet.place(relx=0.40, rely=0.49)
tab_view_label_botnet.place_forget()
label_Label_b = ctk.CTkLabel(master=tab_view_label_botnet, justify=ctk.LEFT, text="BotNet", font=("Roboto", 14))
label_Label_b.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_scan = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_scan.place(relx=0.40, rely=0.615)
tab_view_label_scan.place_forget()
label_Label_s = ctk.CTkLabel(master=tab_view_label_scan, justify=ctk.LEFT, text="PortScan", font=("Roboto", 14))
label_Label_s.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_heart = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_heart.place(relx=0.40, rely=0.74)
tab_view_label_heart.place_forget()
label_Label_h = ctk.CTkLabel(master=tab_view_label_heart, justify=ctk.LEFT, text="Heartbleed", font=("Roboto", 14))
label_Label_h.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_patator = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_patator.place(relx=0.40, rely=0.86)
tab_view_label_patator.place_forget()
label_Label_p = ctk.CTkLabel(master=tab_view_label_patator, justify=ctk.LEFT, text="Patator", font=("Roboto", 14))
label_Label_p.place(relx=0.50, rely=0.60, anchor ="center")

def display2():
    tab_view_label_0.place(relx=0.78, rely=0.00)
    tab_view_label_1.place(relx=0.78, rely=0.125)
    tab_view_label_2.place(relx=0.78, rely=0.245)
    tab_view_label_3.place(relx=0.78, rely=0.365)
    tab_view_label_4.place(relx=0.78, rely=0.49)
    tab_view_label_5.place(relx=0.78, rely=0.615)
    tab_view_label_6.place(relx=0.78, rely=0.74)
    tab_view_label_7.place(relx=0.78, rely=0.86)


button_map_label2 = ctk.CTkButton(master=panel6, width=0, height=80, corner_radius=50, text="",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 12),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=display2)
button_map_label2.place(relx=0.68, rely=0.50, anchor="center")
button_map_label2.place_forget()

tab_view_label_0 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_0.place(relx=0.78, rely=0.00)
tab_view_label_0.place_forget()
label_Label_n = ctk.CTkLabel(master=tab_view_label_0, justify=ctk.LEFT, text="0", font=("Roboto",16))
label_Label_n.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_1 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_1.place(relx=0.78, rely=0.125)
tab_view_label_1.place_forget()
label_Label_d = ctk.CTkLabel(master=tab_view_label_1, justify=ctk.LEFT, text="1", font=("Roboto", 16))
label_Label_d.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_2 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_2.place(relx=0.78, rely=0.245)
tab_view_label_2.place_forget()
label_Label_dd = ctk.CTkLabel(master=tab_view_label_2, justify=ctk.LEFT, text="2", font=("Roboto", 16))
label_Label_dd.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_3 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_3.place(relx=0.78, rely=0.365)
tab_view_label_3.place_forget()
label_Label_w = ctk.CTkLabel(master=tab_view_label_3, justify=ctk.LEFT, text="3", font=("Roboto", 16))
label_Label_w.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_4 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_4.place(relx=0.78, rely=0.49)
tab_view_label_4.place_forget()
label_Label_b = ctk.CTkLabel(master=tab_view_label_4, justify=ctk.LEFT, text="4", font=("Roboto", 16))
label_Label_b.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_5 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_5.place(relx=0.78, rely=0.615)
tab_view_label_5.place_forget()
label_Label_s = ctk.CTkLabel(master=tab_view_label_5, justify=ctk.LEFT, text="5", font=("Roboto", 16))
label_Label_s.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_6 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_6.place(relx=0.78, rely=0.74)
tab_view_label_6.place_forget()
label_Label_h = ctk.CTkLabel(master=tab_view_label_6, justify=ctk.LEFT, text="6", font=("Roboto", 14))
label_Label_h.place(relx=0.50, rely=0.60, anchor ="center")

tab_view_label_7 = ctk.CTkTabview(panel6, corner_radius=15, fg_color="gray30", width=150, height=80)
tab_view_label_7.place(relx=0.78, rely=0.86)
tab_view_label_7.place_forget()
label_Label_p = ctk.CTkLabel(master=tab_view_label_7, justify=ctk.LEFT, text="7", font=("Roboto", 14))
label_Label_p.place(relx=0.50, rely=0.60, anchor ="center")



#create navigation frame
panel7 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel7.place(x = 220, y = 15)
panel7.place_forget()

def balance():
    global bal
    if bal == 0 :
        entry_b_value = entry_b.get()
        if entry_b_value == "":
            entry_b.configure(placeholder_text="Balance", placeholder_text_color="red", border_color="red")
            print("entry_b is empty.")
        else:
            try:
                print("entry_b_value is a valid integer between 1 and 100.")
                val = int(entry_b_value) * 8
                for item in range(len(numbers)):
                    class_item = classA[item]
                    table_bal.insert("", tk.END, values=(class_item, entry_b_value), tags=mytag)
                table_bal.place(relx=0.5, rely=0.0)
                label_Label_b.configure(text="("+str(val)+", 31)")
                label_Label_b.place(relx=0.89, rely=0.45, anchor ="center")
                entry_b.configure(placeholder_text_color="gray", border_color="gray", text_color="gray")
                button_b.focus_set()
                bal = 1
            except ValueError:
                print("entry_b_value is not an integer.")
                entry_b.configure(placeholder_text="Balance", text_color="red", border_color="red")


bal =0
numbers = [1589689, 176302, 111086, 89635, 9689, 1554, 1352, 8]
classA = ["Normal", "Dos", "DDos", "Web Attack", "Botnet", "Scan", "Heartbleed", "Patator"]
# Create the tab view
tab_view_balancing = ctk.CTkTabview(panel7, corner_radius=15, fg_color="gray11", width=815, height=200)
tab_view_balancing.place(relx=0.02, rely=0.02)
tab_bal = tab_view_balancing.add("Data Balancing")
entry_b = ctk.CTkEntry(master=tab_bal, placeholder_text="Balance")
entry_b.place(relx=0.02, rely=0.355)
button_b = ctk.CTkButton(master=tab_bal, width=0, height=60, corner_radius=20, text="Balance",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 16),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=balance)
button_b.place(relx=0.23, rely=0.23)
table_bal = ttk.Treeview(tab_bal, columns=(' Source Port', ' Destination Port'), show = '', height=8)
table_bal.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table_bal["columns"]:
    table_bal.column(column, width=130)
table_bal.pack_forget()
label_Label_b = ctk.CTkLabel(master=tab_bal, justify=ctk.LEFT, font=("Roboto", 20))
label_Label_b.place_forget()

def over():
    global ov
    global num
    if ov == 0 :
        entry_b_value = entry_o.get()
        if entry_b_value == "" or int(entry_b_value) <= 9689:
            entry_o.configure(placeholder_text="Over Sample", placeholder_text_color="red", border_color="red")
            print("entry_o is empty.")
        else:
            try:
                val = 1966712 + ( int(entry_b_value) * 4 )
                all = []
                for i in range(len(num)):
                    num[i] = entry_b_value
                all = n + num
                for item in range(len(all)):
                    value = all[item]
                    class_item = classA[item]
                    table_o.insert("", tk.END, values=(class_item, value), tags=mytag)
                table_o.place(relx=0.5, rely=0.0)
                label_Label_o.configure(text="("+str(val)+", 31)")
                label_Label_o.place(relx=0.89, rely=0.45, anchor ="center")
                entry_o.configure(placeholder_text_color="gray", border_color="gray", text_color="gray")
                button_b.focus_set()
                ov = 1
            except ValueError:
                print("entry_b_value is not an integer.")
                entry_o.configure(placeholder_text= "Over Sample", text_color="red", border_color="red")

n = [1589689, 176302, 111086, 89635]
num = [9689, 1554, 1352, 8]
ov = 0
# Create the tab view
tab_view_over = ctk.CTkTabview(panel7, corner_radius=15, fg_color="gray11", width=815, height=200)
tab_view_over.place(relx=0.02, rely=0.335)
tab_o = tab_view_over.add("Data Over Sampling")
entry_o = ctk.CTkEntry(master=tab_o, placeholder_text="Over Sample")
entry_o.place(relx=0.02, rely=0.355)
button_o = ctk.CTkButton(master=tab_o, width=0, height=60, corner_radius=20, text="Over Sample",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 16),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=over)
button_o.place(relx=0.23, rely=0.23)
table_o = ttk.Treeview(tab_o, columns=(' Source Port', ' Destination Port'), show = '', height=8)
table_o.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table_o["columns"]:
    table_o.column(column, width=130)
table_o.pack_forget()
label_Label_o = ctk.CTkLabel(master=tab_o, justify=ctk.LEFT, font=("Roboto", 20))
label_Label_o.place_forget()



def under():
    global un
    global num2
    if un == 0 :
        entry_b_value = entry_u.get()
        if entry_b_value == "" or int(entry_b_value) >= 89635:
            entry_u.configure(placeholder_text="Under Sample", placeholder_text_color="red", border_color="red")
        else:
            try:
                val = 12603 + ( int(entry_b_value) * 4 )
                all = []
                for i in range(len(n2)):
                    n2[i] = entry_b_value
                all = n2 + num2
                for item in range(len(all)):
                    value = all[item]
                    class_item = classA[item]
                    table_u.insert("", tk.END, values=(class_item, value), tags=mytag)
                table_u.place(relx=0.5, rely=0.0)
                label_Label_u.configure(text="("+str(val)+", 31)")
                label_Label_u.place(relx=0.89, rely=0.45, anchor ="center")
                entry_u.configure(placeholder_text_color="gray", border_color="gray", text_color="gray")
                button_b.focus_set()
                un = 1
            except ValueError:
                entry_u.configure(placeholder_text= "Under Sample", text_color="red", border_color="red")

n2 = [1589689, 176302, 111086, 89635]
num2 = [9689, 1554,  8, 1352]
un = 0
# Create the tab view
tab_view_under = ctk.CTkTabview(panel7, corner_radius=15, fg_color="gray11", width=815, height=200)
tab_view_under.place(relx=0.02, rely=0.66)
tab_u = tab_view_under.add("Data Under Sampling")
entry_u = ctk.CTkEntry(master=tab_u, placeholder_text="Under Sample")
entry_u.place(relx=0.02, rely=0.355)
button_u = ctk.CTkButton(master=tab_u, width=0, height=60, corner_radius=20, text="Under Sample",
                          fg_color="gray20",hover_color=("gray70", "gray30"), font=("Roboto", 16),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=under)
button_u.place(relx=0.23, rely=0.23)
table_u = ttk.Treeview(tab_u, columns=(' Source Port', ' Destination Port'), show = '', height=8)
table_u.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 12))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in table_u["columns"]:
    table_u.column(column, width=130)
table_u.pack_forget()
label_Label_u = ctk.CTkLabel(master=tab_u, justify=ctk.LEFT, font=("Roboto", 20))
label_Label_u.place_forget()


#create navigation frame
panel8 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel8.place(x = 220, y = 15)
panel8.place_forget()

tab_view_allcsv= ctk.CTkTabview(panel8, corner_radius=15, fg_color="gray11", width=180, height=385)
tab_view_allcsv.place(relx=0.02, rely=0.02)
tab_view_n = tab_view_allcsv.add("N")
tab_view_d = tab_view_allcsv.add("D")
tab_view_dd = tab_view_allcsv.add("Dd")
tab_view_w = tab_view_allcsv.add("W")
tab_view_b = tab_view_allcsv.add("B")
tab_view_s = tab_view_allcsv.add("S")
tab_view_h = tab_view_allcsv.add("H")
tab_view_p = tab_view_allcsv.add("P")

tablen = ttk.Treeview(tab_view_n, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tablen.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tablen["columns"]:
    tablen.column(column, width=80)
# Create a horizontal scrollbar
xscrollbarn = ctk.CTkScrollbar(tab_view_n, orientation='horizontal', command=tablen.xview, width=300)
xscrollbarn.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tablen.configure(xscrollcommand=xscrollbarn.set)
tablen.place(x=0, y=0)

tabled = ttk.Treeview(tab_view_d, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tabled.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tabled["columns"]:
    tabled.column(column, width=80)
# Create a horizontal scrollbar
xscrollbard = ctk.CTkScrollbar(tab_view_d, orientation='horizontal', command=tabled.xview, width=300)
xscrollbard.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tabled.configure(xscrollcommand=xscrollbard.set)
tabled.place(x=0, y=0)

tabledd = ttk.Treeview(tab_view_dd, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tabledd.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tabledd["columns"]:
    tabledd.column(column, width=80)
# Create a horizontal scrollbar
xscrollbardd = ctk.CTkScrollbar(tab_view_dd, orientation='horizontal', command=tabledd.xview, width=300)
xscrollbardd.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tabledd.configure(xscrollcommand=xscrollbardd.set)
tabledd.place(x=0, y=0)

tablew = ttk.Treeview(tab_view_w, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tablew.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tablew["columns"]:
    tablew.column(column, width=80)
# Create a horizontal scrollbar
xscrollbarw = ctk.CTkScrollbar(tab_view_w, orientation='horizontal', command=tablew.xview, width=300)
xscrollbarw.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tablew.configure(xscrollcommand=xscrollbarw.set)
tablew.place(x=0, y=0)

tableb = ttk.Treeview(tab_view_b, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tableb.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tableb["columns"]:
    tableb.column(column, width=80)
# Create a horizontal scrollbar
xscrollbarb = ctk.CTkScrollbar(tab_view_b, orientation='horizontal', command=tableb.xview, width=300)
xscrollbarb.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tableb.configure(xscrollcommand=xscrollbarb.set)
tableb.place(x=0, y=0)

tables = ttk.Treeview(tab_view_s, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tables.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tables["columns"]:
    tables.column(column, width=80)
# Create a horizontal scrollbar
xscrollbars = ctk.CTkScrollbar(tab_view_s, orientation='horizontal', command=tables.xview, width=300)
xscrollbars.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tables.configure(xscrollcommand=xscrollbars.set)
tables.place(x=0, y=0)

tableh = ttk.Treeview(tab_view_h, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tableh.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tableh["columns"]:
    tableh.column(column, width=80)
# Create a horizontal scrollbar
xscrollbars = ctk.CTkScrollbar(tab_view_h, orientation='horizontal', command=tableh.xview, width=300)
xscrollbars.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tableh.configure(xscrollcommand=xscrollbars.set)
tableh.place(x=0, y=0)

tablep = ttk.Treeview(tab_view_p, columns=(' Source Port', ' Destination Port', ' Total Fwd Packets',
       'Total Length of Fwd Packets', ' Fwd Packet Length Max',
       ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
       'Bwd Packet Length Max', ' Bwd Packet Length Min',
       ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow IAT Mean',
       ' Flow IAT Max', ' Fwd IAT Std', ' Bwd Packets/s', ' Max Packet Length',
       ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
       ' PSH Flag Count', ' Average Packet Size', ' Avg Fwd Segment Size',
       ' Avg Bwd Segment Size', ' Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       ' Init_Win_bytes_backward', 'Flow ID_enc', 'Source IP_enc',
       'Destination IP_enc', 'Timestamp_enc', 'Label'), show = '', height=30)
tablep.tag_configure('gray',background="gray20", foreground="white", font=("Roboto", 10))
mytag = 'gray'
# Set the width of all columns to 50 pixels
for column in tablep["columns"]:
    tablep.column(column, width=80)
# Create a horizontal scrollbar
xscrollbarp = ctk.CTkScrollbar(tab_view_p, orientation='horizontal', command=tablep.xview, width=300)
xscrollbarp.place(x=0, y=300)
# Set the xscrollbar to control the table's x-axis
tablep.configure(xscrollcommand=xscrollbarp.set)
tablep.place(x=0, y=0)

def output():
    tab_view_allclasses.place(relx=0.58, rely=0.02)
    button_create.place(relx=0.875, rely=0.50, anchor="center")

button_next = ctk.CTkButton(master=panel8, width=0, height=50, corner_radius=50, text="",
                          fg_color="gray10",hover_color=("gray70", "gray30"), font=("Roboto", 14),
                          border_width=4, border_color="#4c4d52", image=imgarrow, compound="right", command=output)
button_next.place(relx=0.49, rely=0.35, anchor="center")

tab_view_allclasses= ctk.CTkTabview(panel8, corner_radius=15, fg_color="gray11", width=180, height=380)
tab_view_allclasses.place_forget()

tab_view_n = tab_view_allclasses.add("N")
normal_image = ctk.CTkImage(Image.open("normal.png"), size=(150, 300))
label_n= ctk.CTkLabel(master=tab_view_n, text="", image=normal_image)
label_n.place(relx=0.0, rely=0.00)
panelcn = ctk.CTkFrame(tab_view_n, corner_radius=30, width=130, height=60)
panelcn.place(x = 160, y = 80)
label_cn= ctk.CTkLabel(master=panelcn, justify=ctk.LEFT, text="Class Normal", font=("Roboto", 16))
label_cn.place(relx=0.50, rely=0.50, anchor="center")

tab_view_d = tab_view_allclasses.add("D")
ddos_image = ctk.CTkImage(Image.open("dos.png"), size=(150, 300))
label_d= ctk.CTkLabel(master=tab_view_d, text="", image=ddos_image)
label_d.place(relx=0.0, rely=0.00)
panelcd = ctk.CTkFrame(tab_view_d, corner_radius=30, width=130, height=60)
panelcd.place(x = 160, y = 80)
label_cd= ctk.CTkLabel(master=panelcd, justify=ctk.LEFT, text="Class Dos", font=("Roboto", 16))
label_cd.place(relx=0.50, rely=0.50, anchor="center")

tab_view_dd = tab_view_allclasses.add("Dd")
ddos_image = ctk.CTkImage(Image.open("ddos.png"), size=(150, 300))
label_dd= ctk.CTkLabel(master=tab_view_dd, text="", image=ddos_image)
label_dd.place(relx=0.0, rely=0.00)
panelcdd = ctk.CTkFrame(tab_view_dd, corner_radius=30, width=130, height=60)
panelcdd.place(x = 160, y = 80)
label_cdd= ctk.CTkLabel(master=panelcdd, justify=ctk.LEFT, text="Class DDos", font=("Roboto", 16))
label_cdd.place(relx=0.50, rely=0.50, anchor="center")

tab_view_w = tab_view_allclasses.add("W")
ddos_image = ctk.CTkImage(Image.open("web.png"), size=(150, 300))
label_w= ctk.CTkLabel(master=tab_view_w, text="", image=ddos_image)
label_w.place(relx=0.0, rely=0.00)
panelcw = ctk.CTkFrame(tab_view_w, corner_radius=30, width=140, height=60)
panelcw.place(x = 160, y = 80)
label_cw= ctk.CTkLabel(master=panelcw, justify=ctk.LEFT, text="Class Web Attack", font=("Roboto", 16))
label_cw.place(relx=0.50, rely=0.50, anchor="center")

tab_view_b = tab_view_allclasses.add("B")
ddos_image = ctk.CTkImage(Image.open("bot.png"), size=(150, 300))
label_b= ctk.CTkLabel(master=tab_view_b, text="", image=ddos_image)
label_b.place(relx=0.0, rely=0.00)
panelcb = ctk.CTkFrame(tab_view_b, corner_radius=30, width=130, height=60)
panelcb.place(x = 160, y = 80)
label_cb= ctk.CTkLabel(master=panelcb, justify=ctk.LEFT, text="Class Botnet", font=("Roboto", 16))
label_cb.place(relx=0.50, rely=0.50, anchor="center")

tab_view_s = tab_view_allclasses.add("S")
ddos_image = ctk.CTkImage(Image.open("scan.png"), size=(150, 300))
label_s= ctk.CTkLabel(master=tab_view_s, text="", image=ddos_image)
label_s.place(relx=0.0, rely=0.00)
panelcs = ctk.CTkFrame(tab_view_s, corner_radius=30, width=130, height=60)
panelcs.place(x = 160, y = 80)
label_cs= ctk.CTkLabel(master=panelcs, justify=ctk.LEFT, text="Class Scan", font=("Roboto", 16))
label_cs.place(relx=0.50, rely=0.50, anchor="center")

tab_view_h = tab_view_allclasses.add("H")
ddos_image = ctk.CTkImage(Image.open("heart.png"), size=(150, 300))
label_h= ctk.CTkLabel(master=tab_view_h, text="", image=ddos_image)
label_h.place(relx=0.0, rely=0.00)
panelch = ctk.CTkFrame(tab_view_h, corner_radius=30, width=140, height=60)
panelch.place(x = 160, y = 80)
label_ch= ctk.CTkLabel(master=panelch, justify=ctk.LEFT, text="Class Heartbleed", font=("Roboto", 16))
label_ch.place(relx=0.50, rely=0.50, anchor="center")

tab_view_p = tab_view_allclasses.add("P")
ddos_image = ctk.CTkImage(Image.open("patator.png"), size=(150, 300))
label_p= ctk.CTkLabel(master=tab_view_p, text="", image=ddos_image)
label_p.place(relx=0.0, rely=0.00)
panelcp = ctk.CTkFrame(tab_view_p, corner_radius=30, width=130, height=60)
panelcp.place(x = 160, y = 80)
label_cp= ctk.CTkLabel(master=panelcp, justify=ctk.LEFT, text="Class patator", font=("Roboto", 16))
label_cp.place(relx=0.50, rely=0.50, anchor="center")


def output2():
    
    def analysis_thread():
        label_Label_patches.place(relx=0.055, rely=0.40, anchor ="center")
        label_Label_.place(relx=0.05, rely=0.85, anchor ="center")

        label_p_n.place(relx=0.16, rely=0.42, anchor ="center")
        panel_p_n.place(x = 95, y = 190)
        time.sleep(0.5)
        panel_p_d.place(x = 182, y = 190)
        label_p_d.place(relx=0.27, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_dd.place(x = 269, y = 190)
        label_p_dd.place(relx=0.38, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_w.place(x = 356, y = 190)
        label_p_w.place(relx=0.49, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_b.place(x = 443, y = 190)
        label_p_b.place(relx=0.60, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_s.place(x = 530, y = 190)
        label_p_s.place(relx=0.71, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_h.place(x = 617, y = 190)
        label_p_h.place(relx=0.82, rely=0.42, anchor ="center")
        time.sleep(0.5)
        panel_p_p.place(x = 704, y = 190)
        label_p_p.place(relx=0.93, rely=0.42, anchor ="center")

    # Create and start the analysis thread
    thread = threading.Thread(target=analysis_thread)
    thread.start()

    


button_create = ctk.CTkButton(master=panel8, width=0, height=50, corner_radius=50, text="Create Patches",
                          fg_color="gray10",hover_color=("gray70", "gray30"), font=("Roboto", 14),
                          border_width=4, border_color="#4c4d52", command=output2)
button_create.place_forget()

tab_view_patches= ctk.CTkTabview(panel8, corner_radius=15, fg_color="gray11", width=790, height=250)
tab_view_patches.place(relx=0.03, rely=0.60)







label_Label_patches = ctk.CTkLabel(master=tab_view_patches, justify=ctk.LEFT, text="Patches:", font=("Roboto",16))
label_Label_patches.pack_forget()

label_Label_ = ctk.CTkLabel(master=tab_view_patches, justify=ctk.LEFT, text="Labels:", font=("Roboto",16))
label_Label_.pack_forget()




normal_image = ctk.CTkImage(Image.open("patch_4/patch_normal_36.png"), size=(60, 150))
label_p_n= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_n = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Ln = ctk.CTkLabel(master=panel_p_n, justify=ctk.LEFT, text="0", font=("Roboto", 18))
label_Ln.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_n.place_forget()
label_p_n.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_dos_144.png"), size=(60, 150))
label_p_d= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_d = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Ld = ctk.CTkLabel(master=panel_p_d, justify=ctk.LEFT, text="1", font=("Roboto", 18))
label_Ld.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_d.place_forget()
label_p_d.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_DDos_57.png"), size=(60, 150))
label_p_dd= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_dd = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Ldd = ctk.CTkLabel(master=panel_p_dd, justify=ctk.LEFT, text="2", font=("Roboto", 18))
label_Ldd.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_dd.place_forget()
label_p_dd.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_web_attack_193.png"), size=(60, 150))
label_p_w= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_w = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Lw = ctk.CTkLabel(master=panel_p_w, justify=ctk.LEFT, text="3", font=("Roboto", 18))
label_Lw.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_w.place_forget()
label_p_w.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_botnet_111.png"), size=(60, 150))
label_p_b= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_b = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Lb = ctk.CTkLabel(master=panel_p_b, justify=ctk.LEFT, text="4", font=("Roboto", 18))
label_Lb.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_b.place_forget()
label_p_b.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_scan_63.png"), size=(60, 150))
label_p_s= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_s = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Ls = ctk.CTkLabel(master=panel_p_s, justify=ctk.LEFT, text="5", font=("Roboto", 18))
label_Ls.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_s.place_forget()
label_p_s.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_heartbleed_73.png"), size=(60, 150))
label_p_h= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_h = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Lh = ctk.CTkLabel(master=panel_p_h, justify=ctk.LEFT, text="6", font=("Roboto", 18))
label_Lh.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_h.place_forget()
label_p_h.place_forget()


normal_image = ctk.CTkImage(Image.open("patch_4/patch_patator_11.png"), size=(60, 150))
label_p_p= ctk.CTkLabel(master=tab_view_patches, text="", image=normal_image)
panel_p_p = ctk.CTkFrame(tab_view_patches, corner_radius=10, width=60, height=50, fg_color="gray30",)
label_Lp = ctk.CTkLabel(master=panel_p_p, justify=ctk.LEFT, text="7", font=("Roboto", 18))
label_Lp.place(relx=0.50, rely=0.50, anchor ="center")
panel_p_p.place_forget()
label_p_p.place_forget()




#create navigation frame DETECTION
panel1 = ctk.CTkFrame(window, corner_radius=20, width=845, height=660)
panel1.place(x = 220, y = 15)
panel1.pack_propagate(False)

# Load the trained model
# Register the custom layer
tf.keras.utils.get_custom_objects()["Patches"] = Patches

tf.keras.utils.get_custom_objects()["PatchEncoder"] = PatchEncoder

new_model = load_model("model_cic_400k_31_col_2_99_94.h5")

def start_analysis():
    global tour

    progressbar_1.configure(mode="indeterminate")
    progressbar_1.start()
    button_start.configure(width=300, text="Scanning...", border_color="#4c4d52",
                            font=("Roboto", 25), fg_color="#1b3864",
                              hover_color=("gray70", "gray30"), border_width=10 )
    tab_view_label_acc.place_forget()
    tab_view_label_loss.place_forget()
    label_Label_r.place_forget()
    label_Label_l.place_forget()

    def analysis_thread():

        textbox.delete("1.0", "end")

        path = 'patch_int/'
        images = []

        for filename in os.listdir(path):
            if filename.endswith('.jpeg') or filename.endswith('.png'):
                img = Image.open(os.path.join(path, filename))
                images_arr_test = np.array(img)
                images.append(images_arr_test)

        images_arr_test = np.array(images)

        

        labels_test = pd.read_csv("labels_int/labels_int.csv", header=None)
        #labels_test = pd.read_csv("labels_tst_80/labels_80.csv", header=None)

        # Convert `labels_test` to a numpy array
        labels_test2 = labels_test.values
        
        # Evaluate the model
        results = new_model.evaluate(images_arr_test, labels_test, verbose=0)
        loss = results[0]
        acc = results[1]
        print(images_arr_test.shape)

        # Predict labels for each image
        predictions = new_model.predict(images_arr_test)
        predicted_labels = np.argmax(predictions, axis=1)

        print("Predictions:")
        for i in range(len(images_arr_test)):
            #print("Image ", i+1, " - True Label: ",labels_test2[i], " Predicted Label: ", predicted_labels[i])
            prediction = "Image {}\t -\t True Label: {}\t -\t    Predicted Label: {}".format(i+1, labels_test2[i], predicted_labels[i])
            print(prediction)
            textbox.insert(tk.END, prediction+"\n")
            time.sleep(0.5)

        print("Evaluation results:")
        print("Loss: ", loss)
        print("Accuracy: ", acc)

        label_Label_r.configure(text=("Accuracy: " + str(int(acc * 100)) + "%"))
        formatted_loss = "{:.9f}".format(loss)
        label_Label_l.configure(text=("Loss: "+str(formatted_loss)))

        tab_view_label_acc.place(relx=0.13, rely=0.2)
        tab_view_label_loss.place(relx=0.13, rely=0.65)

        label_Label_l.place(relx=0.50, rely=0.50, anchor ="center")
        label_Label_r.place(relx=0.50, rely=0.50, anchor ="center")
            
        # Stop the progress bar and update the button text
        progressbar_1.configure(mode="determinate")
        progressbar_1.set(100)
        progressbar_1.stop()
        button_start.configure(width=300,  text="Scan Complete", border_color="green",font=("Roboto", 25), fg_color="#1b3864",
                              hover_color=("gray70", "gray30"), border_width=10 )
    
    # Create and start the analysis thread
    thread = threading.Thread(target=analysis_thread)
    thread.start()


button_start = ctk.CTkButton(master=panel1, width=300, height=250, corner_radius=100,
                              text="Start Scanning",font=("Roboto", 25), fg_color="#1b3864",
                              hover_color=("gray70", "gray30"), command=start_analysis, border_width=10,
                              border_color="#4c4d52")
button_start.place(relx=0.5, rely=0.28, anchor="center")

progressbar_1 = ctk.CTkProgressBar(panel1, width=500)
progressbar_1.place(relx=0.5, rely=0.55, anchor="center")
progressbar_1.set(0)

tab_view_result= ctk.CTkTabview(panel1, corner_radius=15, fg_color="gray11", width=600, height=230)
tab_view_result.place(relx=0.02, rely=0.63)

# create textbox
textbox = ctk.CTkTextbox(tab_view_result, width=500, height=190, font=("Roboto", 18))
textbox.place(relx=0.5, rely=0.55, anchor="center")

tab_view_res= ctk.CTkTabview(panel1, corner_radius=15, fg_color="gray11", width=200, height=230)
tab_view_res.place(relx=0.745, rely=0.63)

tab_view_label_acc = ctk.CTkFrame(tab_view_res, corner_radius=15, fg_color="gray30", width=150, height=60)
tab_view_label_acc.pack_forget()
label_Label_r = ctk.CTkLabel(master=tab_view_label_acc, justify=ctk.LEFT, text="", font=("Roboto",16))
label_Label_r.pack_forget()

tab_view_label_loss = ctk.CTkFrame(tab_view_res, corner_radius=15, fg_color="gray30", width=150, height=60)
tab_view_label_loss.pack_forget()
label_Label_l = ctk.CTkLabel(master=tab_view_label_loss, justify=ctk.LEFT, text="", font=("Roboto",16))
label_Label_l.pack_forget()

# Run the main event loop
window.mainloop()
