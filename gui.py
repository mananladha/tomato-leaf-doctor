import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import pandas 

class TomatoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Leaf Doctor")

        self.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        self.model_files = ['leaf_model.keras', 'leaf_model.h5']
        self.class_names = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold','powdery_mildew','Septoria_leaf_spot', 'Spidermites Two-spotted_spider_mite', 'Target_Spot','Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy']
        
        self.model = None
        self.load_model()
        
        if not self.model:
            messagebox.showerror("Error",
                "No valid model found.\n\n"
                "Please:\n"
                "1. Run new.py first\n"
                "2. Make sure training completes\n"
                "3. Check dataset has enough images")
            self.root.destroy()
            return
            
        self.required_size = self.determine_input_size()
        self.create_widgets()
    
    def load_model(self):
        for model_file in self.model_files:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file)
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                try:
                    self.model = tf.keras.models.load_model(model_path)
                    print(f"Successfully loaded {model_file}")
                    return
                except Exception as e:
                    print(f"Error loading {model_file}: {str(e)}")
                    continue
    
    def determine_input_size(self):
        try:
            
            if hasattr(self.model, 'input'):
                return self.model.input.shape[1:3]  
            
            if len(self.model.layers) > 0 and hasattr(self.model.layers[0], 'input_shape'):
                if self.model.layers[0].input_shape is not None:
                    return self.model.layers[0].input_shape[1:3]
            
            return (224, 224)
            
        except Exception as e:
            print(f"Warning: Could not determine input size, using default. Error: {str(e)}")
            return (224, 224)
    
    def create_widgets(self):
        tk.Label(self.root, text="Tomato Leaf Disease Detector", 
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        self.img_label = tk.Label(self.root)
        self.img_label.pack()
        
        tk.Button(
            self.root, 
            text="Upload Leaf Image", 
            command=self.process_image,
            font=('Arial', 12),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        ).pack(pady=20)
        
        self.result_label = tk.Label(self.root, font=('Arial', 14))
        self.result_label.pack()
        
        self.info_label = tk.Label(self.root, text="", fg='blue')
        self.info_label.pack()
    
    def process_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not filepath:
            return
            
        try:
            img = Image.open(filepath)
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(display_img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
            
            if self.check_image_in_dataset(filepath):
                self.info_label.config(text="This image exists in the dataset", fg='green')
            else:
                self.info_label.config(text="New image detected!", fg='orange')
                if messagebox.askyesno("New Image", "Add this image to dataset?"):
                    self.add_to_dataset(filepath)
            
            self.make_prediction(img)
            
        except Exception as e:
            messagebox.showerror("Error", f"Image processing failed: {str(e)}")
    
    def check_image_in_dataset(self, image_path):
        img_name = os.path.basename(image_path)
        for class_name in self.class_names:
            class_dir = os.path.join(self.dataset_path, class_name)
            if os.path.exists(os.path.join(class_dir, img_name)):
                return True
        return False
    
    def add_to_dataset(self, image_path):
        class_name = simpledialog.askstring(
            "Select Disease Type",
            "Enter the disease class:",
            parent=self.root
        )
        
        if class_name and class_name in self.class_names:
            dest_dir = os.path.join(self.dataset_path, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            try:
                shutil.copy2(image_path, dest_dir)
                messagebox.showinfo("Success", f"Image added to {class_name} class!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add image: {str(e)}")
        else:
            messagebox.showwarning("Invalid", "Please enter a valid class name")
    
    def make_prediction(self, img):
        try:
            img = img.convert('RGB').resize(self.required_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = self.model.predict(img_array)
            class_idx = np.argmax(pred[0])
            confidence = pred[0][class_idx] * 100

            self.result_label.config(
                text=f"Diagnosis: {self.class_names[class_idx]}\nConfidence: {confidence:.1f}%",
                fg='green' if confidence > 75 else 'orange'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoApp(root)
    root.mainloop()