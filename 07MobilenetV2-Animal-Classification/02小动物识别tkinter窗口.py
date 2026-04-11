import tkinter as tk
from tkinter import filedialog
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image,ImageTk

animal=['兔子', '狗', '猪', '猫', '猴子', '老虎', '蛇', '马', '鸡']
load_model=models.load_model('mobilenetv2-animal-classification.keras')
root = tk.Tk()
root.title('小动物识别')
root.geometry('500x500')
def load_image():
    img_path=filedialog.askopenfilename(title='选择图片',filetypes=[('PNG','*.png'),('JPG','*.jpg'),('JPEG','*.jpeg')])
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0)
    rates=load_model.predict(img_array)
    result=animal[np.argmax(rates)]
    label1.config(text=f'预测的结果是：{result}')
    image_=Image.open(img_path).resize((300,300))
    imagetk=ImageTk.PhotoImage(image_)
    label2.config(image=imagetk)
    label2.image=imagetk

button=tk.Button(root,text='请上传相关图片',command=load_image)
button.pack(pady=30)
label1=tk.Label(root,text='预测的结果是：')
label1.pack()
label2=tk.Label(root)
label2.pack()
root.mainloop()