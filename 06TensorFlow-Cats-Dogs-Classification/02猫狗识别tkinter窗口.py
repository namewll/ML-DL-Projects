import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
load_model=models.load_model('cat_dog_model.keras')

root = tk.Tk()
root.title('猫狗图片预测')
root.geometry('500x500')
def load_image():
    img_path = filedialog.askopenfilename(title='请选择相关图片',filetypes=[('PNG Files','*.png'),('JPG Files','*.jpg'),('JPEG Files','*.jpeg')])
    if img_path:
        img=image.load_img(img_path,target_size=(150,150))
        img_array=image.img_to_array(img)
        img_array=img_array.reshape((1,150,150,3))
        img_array=img_array/255.0
        out_put=load_model.predict(img_array)
        result='猫' if out_put[0]<0.5 else '狗'
        label1.config(text=result)
        img_open=Image.open(img_path).resize((300,300))
        img_update=ImageTk.PhotoImage(img_open)
        label2.config(image=img_update)
        label2.image=img_update

button=tk.Button(root,text='上传图片进行预测',command=load_image)
button.pack(pady=50)
label1=tk.Label(root,text='预测结果是:')
label1.pack()
label2=tk.Label(root)
label2.pack(pady=20)
root.mainloop()