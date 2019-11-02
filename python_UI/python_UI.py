"""==============================================================================
    Author: Nom Phan
    Description: Python UI that dislays keyboard, sentences suggestions
================================================================================="""

import tkinter
from tkinter import *
from tkinter import font

from gtts import gTTS
from pygame import mixer

kb = tkinter.Tk()

buttons = [
    '~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', 'DEL',
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '\\', '7', '8', '9', 'BACK',
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '[', ']', '4', '5', '6', 'SHIFT',
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '?', '/', '1', '2', '3', 'SPACE'
]


def select(value):
    """

    :param value:
    :return:
    """
    if value == "BACK":
        # allText = entry.get()[:-1]
        # entry.delete(0, tkinter,END)
        # entry.insert(0,allText)

        entry.delete(
            len(entry.get()) - 1,
            tkinter.END
        )
    elif value == "DEL":
        entry.delete(
            0,
            last=len(entry.get())
        )
    elif value == "SPACE":
        entry.insert(
            tkinter.END,
            ' '
        )
    elif value == " Tab ":
        entry.insert(
            tkinter.END,
            '    '
        )
    else:
        entry.insert(
            tkinter.END,
            value)


helv36 = font.Font(family='Helvetica', size=10)
helv30 = font.Font(family='Helvetica', size=20, weight="bold")
helv20 = font.Font(family='Helvetica', size=10, weight="bold")


def keyboard_display():
    """

    :return:
    """
    varRow = 10
    var_column = 0

    for button in buttons:

        command = lambda x=button: select(x)

        if button == "SPACE" or button == "SHIFT" or button == "BACK" or button == "DEL":
            tkinter.Button(
                kb,
                text=button,
                font=helv36,
                height=1,
                width=8,
                bg="#3c4987",
                fg="#000000",
                activebackground="#ffffff",
                activeforeground="#3c4987",
                relief='raised',
                padx=1,
                pady=1,
                bd=1,
                command=command
            ).grid(row=varRow, column=var_column)
        else:
            tkinter.Button(
                kb,
                text=button,
                font=helv36,
                height=1,
                width=10,
                bg="#3c4987",
                fg="#000000",
                activebackground="#ffffff",
                activeforeground="#3c4987",
                relief='raised',
                padx=1,
                pady=1,
                bd=1,
                command=command
            ).grid(row=varRow, column=var_column)

        var_column += 1

        if var_column > 14 and varRow == 10:
            var_column = 0
            varRow += 1
        if var_column > 14 and varRow == 11:
            var_column = 0
            varRow += 1
        if var_column > 14 and varRow == 12:
            var_column = 0
            varRow += 1


def text_to_speech(my_text):
    """

    :param my_text:
    :return:
    """
    language = 'en'
    output = gTTS(text=my_text, lang=language, slow=False)
    output.save("output.mp3")

    mixer.init()
    mixer.music.load("output.mp3")
    mixer.music.play()

    entry.delete(
        0,
        last=len(entry.get())
    )


def main():
    """

    :return:
    """
    kb.title("MIND READER")

    label1 = Label(
        kb,
        text='MIND READER',
        font=helv30
    ).grid(row=0, columnspan=15)

    global entry
    entry = Entry(
        kb,
        width=60,
        justify="left"
    ).grid(row=6, column=1, columnspan=12)

    button = Button(
        kb, text="SPEAK",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: text_to_speech(entry.get())
    ).grid(row=6, column=10, columnspan=5)

    button = Button(
        kb,
        text="I want to eat",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "I want to eat ")
    ).grid(row=8, columnspan=5)

    button = Button(
        kb,
        text="I want to sleep",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "I want to sleep ")
    ).grid(row=8, column=5, columnspan=5)

    button = Button(
        kb,
        text="Good morning",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "Good morning ")
    ).grid(row=8, column=10, columnspan=5)

    button = Button(
        kb,
        text="Help me get up",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "Help me get up ")
    ).grid(row=9, columnspan=5)

    button = Button(
        kb,
        text="Excuse me",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "Excuse me ")
    ).grid(row=9, column=5, columnspan=5)

    button = Button(
        kb,
        text="How are you",
        font=helv20,
        width=55,
        height=4,
        fg="red",
        command=lambda: entry.insert(tkinter.END, "How are you ")
    ).grid(row=9, column=10, columnspan=5)

    keyboard_display()

    kb.mainloop()


main()
