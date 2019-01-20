# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:14:49 2019

@author: Suraj Kumar
"""

from tkinter import Label,Tk,Canvas,LEFT
from tkinter.ttk import Button
from PIL import Image,ImageDraw
from keras.models import load_model
import numpy as np

main_window=Tk()
main_window.title("Character Recognition Demo - by SJ-Ray")
main_window.iconbitmap(r'ai.ico')

index2label={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

#Canvas Code Section   
canvas_width=600
canvas_height=300
cnvs_title=Label(main_window,height=1,text="Canvas Area",highlightthickness=1,borderwidth=0,background='#A9A9A9')
cnvs_title.grid(row=0,column=0,sticky='EW')
cnvs = Canvas(main_window,width=canvas_width,height=canvas_height,background='#FFFFFF')
cnvs.grid(row=1,column=0,sticky='NEWS')

image = Image.new("L", (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(image)

def getImage():
    img=np.array(image)
    x0,x1=(0,0)
    y0,y1=(0,0)
    x_flag=True
    y_flag=True
    
    for i in range(0,img.shape[1]):
        if img[:,i].any():
            if x_flag:
                x0=i
                x_flag=False
            else:
                x1=i
                
        if i<img.shape[0]:
            if img[i,:].any():
                if y_flag:
                    y0=i
                    y_flag=False
                else:
                    y1=i
                    
    pad=10
    img2=img[y0-pad:y1+pad,x0-pad:x1+pad]
    pil_image=Image.fromarray(img2)
    pil_image=np.array(pil_image.resize((28,28)))
    pil_image_reshaped=pil_image.reshape(1, 1, 28, 28).astype('float32')/255
    probab=model.predict_proba([pil_image_reshaped])[0]
    probab_dict=dict(zip(index2label.values(),probab))
    sorted_probab_dict=sorted(probab_dict.items(), key=lambda kv: kv[1],reverse=True)
    to_display="Predictions:-\n\n\t"+"\n\t".join([ key+' : '+str(round(value*100,2))+'%' for key,value in sorted_probab_dict[0:10]])
    result_area.configure(text=to_display)
 
def paint( event ):
   black_color = "#000000"
   r=3
   x1, y1 = ( event.x - r ), ( event.y - r )
   x2, y2 = ( event.x + r ), ( event.y + r )
   cnvs.create_oval( x1, y1, x2, y2, fill = black_color)
   draw.ellipse([x1,y1,x2,y2], fill=255)

def clear( event ):
    cnvs.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height],0)
    result_area.configure(text="Predictions\n\n")
    
cnvs.bind( "<B1-Motion>", paint )

##Canvas Controls
clear_btn=Button(text='Clear')
clear_btn.grid(row=2,column=0,sticky='EW')
clear_btn.bind("<1>",clear)
run_btn=Button(text='Run',command=getImage)
run_btn.grid(row=2,column=1,sticky='EW')
#Result Code Section
text_title=Label(main_window,height=1,text='Predictions',highlightthickness=1,borderwidth=0,background='#A9A9A9')
text_title.grid(row=0,column=1,sticky='EW')
result_area=Label(main_window,width=50,text='Predictions\n\n',foreground='#20C20E',highlightthickness=1,borderwidth=0,background='#000000')
result_area.config(justify=LEFT,pady=4,padx=10,anchor='w',font=("Courier", 15))
result_area.grid(row=1,column=1,sticky='NS')                  

model=load_model("alpha_numeric_recognizer.hdf5")

main_window.mainloop()