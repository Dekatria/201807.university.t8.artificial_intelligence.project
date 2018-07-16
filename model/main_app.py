from tkinter import Tk, Label, Button, Entry, StringVar, filedialog
from PIL import Image, ImageTk
import predict


class VQAGUI:
    def __init__(self, master):
        self.master = master
        master.title("VQA GUI")
        master.geometry("500x500")
        answer1 = StringVar()
        answer2 = StringVar()
        answer3 = StringVar()
        answer4 = StringVar()
        answer5 = StringVar()
        self.answers = [answer1, answer2, answer3, answer4, answer5]
        #        self.label = Label(master, text="This is our first GUI!")
        #        self.label.pack()
        #        self.label.grid(column=1,row=0)

        self.question = Entry(master, width=30)

        self.imageLabel = Label()
        self.imageLabel.grid(row=1, column=1)

        self.browse_button = Button(master, text="Browse...", command=self.askOpenFile)
        self.browse_button.grid(row=1, column=2)

        self.question.grid(row=2, column=1)
        #        self.txt.pack()

        #        self.close_button = Button(master, text="Close", command=master.quit)
        #        self.close_button.pack()

        self.print_button = Button(master, text="Question?", command=self.getAnswers)
        self.print_button.grid(row=2, column=2)

        self.answer_label = Label(master, text="Top 5 responses: ")
        self.answer_label.grid(row=3, column=1)
        self.machine_answer_label1 = Label(master, textvariable=answer1)  # " ".join(self.getAnswers()))

        self.machine_answer_label1.grid(row=4, column=1)
        self.machine_answer_label2 = Label(master, textvariable=answer2)
        self.machine_answer_label2.grid(row=5, column=1)
        self.machine_answer_label3 = Label(master, textvariable=answer3)
        self.machine_answer_label3.grid(row=6, column=1)
        self.machine_answer_label4 = Label(master, textvariable=answer4)
        self.machine_answer_label4.grid(row=7, column=1)
        self.machine_answer_label5 = Label(master, textvariable=answer5)
        self.machine_answer_label5.grid(row=8, column=1)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.grid(row=4, column=3)


    #        self.print_button.pack()
    def getAnswers(self):
        res = []

        #import predict
        #TODO put input img path and question.

        #predict.main(img,question)
        # get answers from model eval forward
        ##### ASK QUESTION HERE

        # for i in range(len(self.answers)):
        # self.answers[i].set(res[i])

        ans_map, answer_probab_tuples = predict.main(self.filename, self.getQuestion())

        for i in range(5):
            res.append(ans_map[answer_probab_tuples[i][1]])
            self.answers[i].set(res[i])
            


    def getQuestion(self):
        print("Asked question: ", self.question.get())

        return self.question.get()

    def askOpenFile(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if self.filename != "":
            self.openImage(self.filename)

    def openImage(self, filename):
        img = Image.open(self.filename)
        basewidth = 300
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        self.image = img.resize((basewidth, hsize), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(self.image)
        self.imageLabel.configure(image=photo)
        self.imageLabel.image = photo  # keep a reference!


app = Tk()
VQAGUI(app)
app.mainloop()