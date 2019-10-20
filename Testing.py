from tkinter import *
from tkinter import font
import tkinter
from gtts import gTTS
import os
from pygame import mixer

kb = tkinter.Tk()

buttons = [
'~','`','!','@','#','$','%','^','&','*','(',')','-','_','DEL',
'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p','\\','7','8','9','BACK',
'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l','[',']','4','5','6'
,'SHIFT',
'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.','?','/','1','2','3','SPACE'
]
def select(value):
    if value == "BACK":
        # allText = entry.get()[:-1]
        # entry.delete(0, tkinter,END)
        # entry.insert(0,allText)

        entry.delete(len(entry.get())-1,tkinter.END)
    elif value == "DEL":
        entry.delete(0, last = len(entry.get()))
    elif value == "SPACE":
        entry.insert(tkinter.END, ' ')
    elif value == " Tab ":
        entry.insert(tkinter.END, '    ')
    else :
        entry.insert(tkinter.END,value)
helv36 = font.Font(family='Helvetica', size=10)
helv30 = font.Font(family='Helvetica', size=20, weight = "bold")
helv20 = font.Font(family='Helvetica', size=10, weight = "bold")
def HosoPop():

    varRow = 10
    varColumn = 0

    for button in buttons:

        command = lambda x=button: select(x)
        
        if button == "SPACE" or button == "SHIFT" or button == "BACK" or button == "DEL":
            tkinter.Button(kb,text= button,font=helv36,height=1,width=8, bg="#3c4987", fg="#000000",
                activebackground = "#ffffff", activeforeground="#3c4987", relief='raised', padx=1,
                pady=1, bd=1,command=command).grid(row=varRow,column=varColumn)
        else:
            tkinter.Button(kb,text= button,font=helv36,height=1,width=10, bg="#3c4987", fg="#000000",
                activebackground = "#ffffff", activeforeground="#3c4987", relief='raised', padx=1,
                pady=1, bd=1,command=command).grid(row=varRow,column=varColumn)


        varColumn +=1

        if varColumn > 14 and varRow == 10:
            varColumn = 0
            varRow+=1
        if varColumn > 14 and varRow == 11:
            varColumn = 0
            varRow+=1
        if varColumn > 14 and varRow == 12:
            varColumn = 0
            varRow+=1

def text_to_speech(my_Text):
    language = 'en'
    output = gTTS(text = my_Text,lang = language,slow = False)
    output.save("output.mp3")
    
    mixer.init()
    mixer.music.load("output.mp3")
    mixer.music.play()

    entry.delete(0, last = len(entry.get()))
    
def main():

    
    kb.title("MIND READER")
 
    label1 = Label(kb,text='MIND READER', font = helv30).grid(row=0, columnspan = 15)
    
    
    
    global entry
    entry = Entry(kb,width= 60, justify = "left")
    entry.grid(row=6,column = 1,columnspan = 12)
    
    button = Button(kb, text="SPEAK", font=helv20, width = 55, height =4, fg = "red", command=lambda: text_to_speech(entry.get()))
    button.grid(row=6, column = 10, columnspan = 5)
    
    
    button = Button(kb, text="I want to eat", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"I want to eat "))
    button.grid(row=8, columnspan = 5)
    
    button = Button(kb, text="I want to sleep", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"I want to sleep "))
    button.grid(row=8, column = 5 ,columnspan = 5)
    
    button = Button(kb, text="Good morning", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"Good morning "))
    button.grid(row=8, column = 10 ,columnspan = 5)
    
    button = Button(kb, text="Help me get up", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"Help me get up "))
    button.grid(row=9, columnspan = 5)
    
    button = Button(kb, text="Excuse me", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"Excuse me "))
    button.grid(row=9, column = 5 ,columnspan = 5)
    
    button = Button(kb, text="How are you", font=helv20, width = 55, height =4, fg = "red", command = lambda: entry.insert(tkinter.END,"How are you "))
    button.grid(row=9, column = 10 ,columnspan = 5)
    
    HosoPop()

    kb.mainloop()
main()
