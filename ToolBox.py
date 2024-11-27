from math import exp, sqrt
from tkinter import (
    HORIZONTAL, NW, Entry, Button, Canvas, Frame, Label, Scale, 
    StringVar, messagebox, Tk, filedialog, PhotoImage
)
from tkinter.messagebox import showinfo
from tkinter.tix import Balloon
from tkinter.ttk import Combobox
import cv2 as cv
import PIL.Image, PIL.ImageTk
import numpy as np
from numpy import asarray
import os
import matplotlib
from matplotlib import pyplot as plt
# from tkinter.tix import *
import statistics
from easyocr import Reader
from ArabicOcr import arabicocr
matplotlib.use('TkAgg')
#####################################################################################################

def Image_Browse():
    global image
    global photo
    global originalImage
    global finalEdit
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),
                                    title="Select image", 
                                    filetypes=(("All Files","*.*"),("JPG File","*.jpg"),("PNG File","*.png")))
    image = PIL.Image.open(fln)
    image = asarray(image)
    if image.shape[1] > 900 or image.shape[0] > 700:
        diag = (805,565)
        image = cv.resize(image,diag, interpolation = cv.INTER_AREA)
    # if image.shape[1] > 900:
    #     image = cv.resize(image,(806,image.shape[1]), interpolation = cv.INTER_AREA)
    # if image.shape[0] > 700:
    #     image = cv.resize(image,(image.shape[0],600), interpolation = cv.INTER_AREA)
    originalImage = image
    finalEdit = image
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def save():
    global finalEdit
    global image
    global photo
    finalEdit = image
    cv.imwrite('save.png',finalEdit)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def saveAs():
    fS = filedialog.asksaveasfilename(initialdir=os.getcwd(),filetypes=(("PNG File", "*.png"),("JPG File", "*.jpg")),
                                    defaultextension=".png", title="Save As")
    cv.imwrite(fS, image)

def saveC(path, image, jpg_quality=None, png_compression=None):
    if jpg_quality:
        cv.imwrite(path, image, [int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
    elif png_compression:
        cv.imwrite(path, image, [int(cv.IMWRITE_PNG_COMPRESSION), png_compression])
    else:
        cv.imwrite(path, image)

def Lossless_compression_SaveAs():
    global image
    global photo
    fS = filedialog.asksaveasfilename(initialdir=os.getcwd(),filetypes=(("JPG File", "*.jpg"),("PNG File", "*.png")),
                                    defaultextension=".jpg", title="Save As")
    saveC(fS,image,jpg_quality=65)

def Lossy_compression_SaveAs():
    global image
    global photo
    fS = filedialog.asksaveasfilename(initialdir=os.getcwd(),filetypes=(("JPG File", "*.jpg"),("PNG File", "*.png")),
                                    defaultextension=".jpg", title="Save As")
    saveC(fS,image,jpg_quality=25)

# def compressorfun():
#     global image
#     my_string = image
#     shape = my_string.shape
#     a = my_string
#     print ("Enetered string is:",my_string)
#     my_string = str(my_string.tolist())  

#     letters = []
#     only_letters = []
#     for letter in my_string:
#         if letter not in letters:
#             frequency = my_string.count(letter)             #frequency of each letter repetition
#             letters.append(frequency)
#             letters.append(letter)
#             only_letters.append(letter)

#     nodes = []
#     while len(letters) > 0:
#         nodes.append(letters[0:2])
#         letters = letters[2:]                               # sorting according to frequency
#     nodes.sort()
#     huffman_tree = []
#     huffman_tree.append(nodes)                             #Make each unique character as a leaf node

#     def combine_nodes(nodes):
#         pos = 0
#         newnode = []
#         if len(nodes) > 1:
#             nodes.sort()
#             nodes[pos].append("1")                       # assigning values 1 and 0
#             nodes[pos+1].append("0")
#             combined_node1 = (nodes[pos] [0] + nodes[pos+1] [0])
#             combined_node2 = (nodes[pos] [1] + nodes[pos+1] [1])  # combining the nodes to generate pathways
#             newnode.append(combined_node1)
#             newnode.append(combined_node2)
#             newnodes=[]
#             newnodes.append(newnode)
#             newnodes = newnodes + nodes[2:]
#             nodes = newnodes
#             huffman_tree.append(nodes)
#             combine_nodes(nodes)
#         return huffman_tree                                     # huffman tree generation

#     newnodes = combine_nodes(nodes)

#     huffman_tree.sort(reverse = True)

#     checklist = []
#     for level in huffman_tree:
#         for node in level:
#             if node not in checklist:
#                 checklist.append(node)
#             else:
#                 level.remove(node)
#     count = 0
#     for level in huffman_tree:
#         count+=1

#     letter_binary = []
#     if len(only_letters) == 1:
#         lettercode = [only_letters[0], "0"]
#         letter_binary.append(lettercode*len(my_string))
#     else:
#         for letter in only_letters:
#             code =""
#             for node in checklist:
#                 if len (node)>2 and letter in node[1]:           #genrating binary code
#                     code = code + node[2]
#             lettercode =[letter,code]
#             letter_binary.append(lettercode)
            
#     for letter in letter_binary:
#         print(letter[0], letter[1])

#     bitstring =""
#     for character in my_string:
#         for item in letter_binary:
#             if character in item:
#                 bitstring = bitstring + item[1]
#     binary ="0b"+bitstring

#     uncompressed_file_size = len(my_string)*7
#     compressed_file_size = len(binary)-2
#     print("Your original file size was", uncompressed_file_size,"bits. The compressed size is:",compressed_file_size)
#     print("This is a saving of ",uncompressed_file_size-compressed_file_size,"bits")
#     output = open("compressed.txt","w+")
#     print("Compressed file generated as compressed.txt")
#     output = open("compressed.txt","w+")
#     print("Decoding.......")
#     output.write(bitstring)

#     bitstring = str(binary[2:])
#     uncompressed_string =""
#     code =""
#     for digit in bitstring:
#         code = code+digit
#         pos=0                                        #iterating and decoding
#         for letter in letter_binary:
#             if code ==letter[1]:
#                 uncompressed_string=uncompressed_string+letter_binary[pos] [0]
#                 code=""
#             pos+=1

#     print("Your UNCOMPRESSED data is:")

#     temp = re.findall(r'\d+', uncompressed_string)
#     res = list(map(int, temp))
#     res = np.array(res)
#     res = res.astype(np.uint8)
#     res = np.reshape(res, shape)
#     cv2.imwrite("./CompressImages/CompressHoffman.jpg",res)
#     if a.all() == res.all():
#         print("Success")
#         messagebox.showinfo(title="compressor Message", message="compressor Done"+"\n"+"image store in same path of project")

def resets():
    global image
    global originalImage
    global finalEdit
    global photo
    global arrOfCanvas
    global photo3,photo4
    global imageTest_1,imageTest_2
    image = originalImage
    finalEdit = originalImage
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    photo3 =  photo4 = photo5 = ''
    imageTest_1 = imageTest_2 = imageTest_3 = ''
    canvasTest_1.create_image(0,0,image=photo3, anchor=NW)
    canvasTest_2.create_image(0,0,image=photo4, anchor=NW)
    arrOfCanvas = 0
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def restore():
    global image
    global photo
    finalEdit = cv.imread("save.png",0)
    image = finalEdit
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def grayScale():
    global image
    global photo
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    getMeanAndStandardDeviation()
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def crop():
    global image
    global photo
    messagebox.showinfo(title="sorry", message="crop not work now")
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0


def mouse_crop(event, x, y, flags, param):
    global image
    oriImage = image.copy()
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv.imshow("Cropped", roi)
            image = roi


def myCrop():
    global image
    global photo
    cv.namedWindow("image")
    cv.setMouseCallback("image", mouse_crop)

    i = image.copy()

    if not cropping:
        cv.imshow("image", image)

    elif cropping:
        cv.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.imshow("Preview Crop", i)
        image = i
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)
        findEmptyCanves(image)
    cv.waitKey(0)
    # close all open windows
    cv.destroyAllWindows()
    applyCrop()

def applyCrop():
    global image
    global photo
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def RotateImgR ():
    global image
    global photo
    image = cv.rotate(image,cv.ROTATE_90_CLOCKWISE)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def RotateImgL ():
    global image
    global photo
    image = cv.rotate(image,cv.ROTATE_90_COUNTERCLOCKWISE)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def translationTop():
    global image
    global photo
    M = np.float32([[1, 0, 0],[0, 1, 10]])
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def translationLeft():
    global image
    global photo
    M = np.float32([[1, 0, 10],[0, 1, 0]])
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def translationRight():
    global image
    global photo
    M = np.float32([[1, 0, -10],[0, 1, 0]])
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def translationBottom():
    global image
    global photo
    M = np.float32([[1, 0, 0],[0, 1, -10]])
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def translationTopLeft():
    translationTop()
    translationLeft()

def translationTopRight():
    translationTop()
    translationRight()

def translationDownLeft():
    translationBottom()
    translationLeft()

def translationDownRight():
    translationBottom()
    translationRight()

def skewingRTransformation():
    global image
    global photo
    point_1 = np.float32([[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]]) 
    point_2 = np.float32([[10, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]])
    M = cv.getAffineTransform(point_1, point_2)
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def skewingLTransformation():
    global image
    global photo
    point_1 = np.float32([[10, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]]) 
    point_2 = np.float32([[0, 0], [0, image.shape[1]], [image.shape[0], image.shape[1]]])
    M = cv.getAffineTransform(point_1, point_2)
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def flip():    
    global image
    global photo
    image = cv.flip(image, 0)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def flipVertical():
    global image
    global photo
    image = cv.flip(image, 1)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def Merge_Image_Browse():
    global Merge_Image
    global photo
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select image", filetypes=(("All Files","*.*"),("JPG File","*.jpg"),("PNG File","*.png")))
    Merge_Image = PIL.Image.open(fln)
    Merge_Image = asarray(Merge_Image)
    if Merge_Image.shape[0] > 600:
        dim = (600,470)
        Merge_Image = cv.resize(Merge_Image,dim, interpolation = cv.INTER_AREA)
    photo = PIL.ImageTk.PhotoImage(Merge_Image = PIL.Image.fromarray(Merge_Image))
    canvas.create_image(0, 0, Merge_Image=photo, anchor=NW)
    findEmptyCanves(image)

def merge():
    global image
    global photo
    global Merge_Image
    #marge with gray scale
    # image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    # Merge_Image = cv.cvtColor(Merge_Image,cv.COLOR_RGB2GRAY)
    resized = cv.resize(Merge_Image,(image.shape[1],image.shape[0]),interpolation=cv.INTER_AREA)
    #another way to marge without function
    # for row in range(image.shape[0]):
    #     for col in range(image.shape[1]):
    #         image[row,col] = image[row,col] * 0.8 + resized[row,col] * 0.2
    image= cv.addWeighted(image,0.8,resized,0.2,0)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def histGraph():
    global image

    plt.plot(cv.calcHist([image],[0],None,[256],[0,256]))
    plt.show()

    #plt.hist(image.ravel(),256,[0,256])
    #plt.show()

    #RGB histGraph
    # color = ('b','g','r')
    # for i,col in enumerate(color):
    #     histr = cv.calcHist([image],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.show()

def equalizeHist():
    global image
    global photo
    image = cv.equalizeHist(image)

    #RGB equalizeHist
    # image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    # image[:,:,0] = cv.equalizeHist(image[:,:,0])
    # image = cv.cvtColor(image, cv.COLOR_YUV2BGR)

    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def negativeTrans():
    global image
    global photo
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            image[row,col] =  255 - image[row,col]
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def LogarithmicTrans():
    global image
    global photo
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(image + 1)
    image = np.array(log_image, dtype = np.uint8)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def PowerTrans():
    global image
    global photo
    if(gammaVal.get() == ''):
        messagebox.showwarning(title="Warning", message="you must enter gamma value he prefers gamma value between 0 and 0.999 or between 2 and 25")
    else:
        image = np.array(255*(image / 255) ** float (gammaVal.get()), dtype = 'uint8')
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

#####################################################################################
#Bit Plane all step

def cov_binary(num):
    binary_num = [int(i) for i in list('{0:0b}'.format(num))]
    for j in range(8 - len(binary_num)):
        binary_num.insert(0,0)        
    return binary_num
def conv_decimal(listt):
    x = 0
    for i in range(8):
        x = x + int(listt[i])*(2**(7-i))
    return x
def discriminate_bit(bit,img):
    global image
    global photo
    row, column= image.shape
    z = np.zeros((row,column),dtype = 'uint8')
    for i in range(row):
        for j in range(column):
            x = cov_binary(img[i][j])
            for k in range(8):
                if k == bit:
                    x[k] = x[k]
                else:
                    x[k] = 0
            x1 = conv_decimal(x)
            z[i][j] = x1
    return z
def Bit_Plane():
    global image
    global photo
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i in range(1,9):
        fig.add_subplot(4,2,i)
        plt.imshow(discriminate_bit(i-1,image), cmap='gray')
        
    plt.show()

#####################################################################################

def Last_Bit_Plane():
    global image
    global photo
    row, column= image.shape
    img = image
    if(Bit_Val.get() == ''):
        val = 128
    elif(int(Bit_Val.get()) <= 2):
        val = 2
    elif(int(Bit_Val.get()) <= 4):
        val = 4
    elif(int(Bit_Val.get()) <= 8):
        val = 8
    elif(int(Bit_Val.get()) <= 16):
        val = 16
    elif(int(Bit_Val.get()) <= 32):
        val = 32
    elif(int(Bit_Val.get()) <= 64):
        val = 64
    elif(int(Bit_Val.get()) > 64):
        val = 128
    image = np.zeros((row,column),dtype = 'uint8')
    for i in range(row):
        for j in range(column):
            if img[i,j]& int(val): 
                image[i,j] = 255
            else: 
                image[i,j] = 0 
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def Gray_level():
    global image
    global photo
    row, column= image.shape
    img = image
    image = np.zeros((row,column),dtype = 'uint8')
    if Max_Gray_Val.get() == '' or Min_Gray_Val.get() == '':
        messagebox.showwarning(title="Warning", message="plz enter max and min")
    else:
        max_range = int(Max_Gray_Val.get())
        min_range = int(Min_Gray_Val.get())
    for i in range(row):
        for j in range(column):
            if img[i,j]>min_range and img[i,j]<max_range: 
                image[i,j] = 255
            else: 
                image[i,j] = img[i-1,j-1] 
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def BlurCombobox(event):
    global image
    global photo
    filters = event.widget.get()
    if filters == 'Blur 3X3':
        kernal_3X3 = np.ones((3,3), np.float32) / 9
        image = cv.filter2D(image, -1, kernal_3X3)
    # elif filters == 'Blur 5X5':
    #     kernal_5X5 = np.ones((5,5), np.float32) / 25
    #     image = cv.filter2D(image, -1, kernal_5X5)
    #     image = cv.GaussianBlur(image,(9,9),0)
    elif filters == 'Gaussian filter':
        image = cv.GaussianBlur(image,(3,3),0)
        # kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
        # image = cv.filter2D(image, -1, kernel)
    # elif filters == 'Gaussian 9X9 blur':
        #image = cv.GaussianBlur(image,(9,9),0)
    elif filters == 'Median filter':
        image = cv.medianBlur(image,9)
    elif filters == 'Bilateral filter':
        image = cv.bilateralFilter(image,10,250,250)
    elif filters == 'pyramidal':
        kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3],
                            [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]], np.float32) / 81
        image = cv.filter2D(image, -1, kernel)
    elif filters == 'circular':
        kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1,1],
                            [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], np.float32) / 21
        image = cv.filter2D(image, -1, kernel)
    elif filters == 'cone':
        kernel = np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1],
                            [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]], np.float32) / 21
        image = cv.filter2D(image, -1, kernel)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def sharpingOne():
    global image
    global photo
    kernal_shearing = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
    image = cv.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def sharpingTwo():
    global image
    global photo
    kernal_shearing = np.array([[1, 1, 1],[1, -7, 1],[1, 1, 1]])
    image = cv.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def sharpingThree():
    global image
    global photo
    kernal_shearing = np.array([[-1,-1,-1,-1,-1],
                                [-1,2,2,2,-1],
                                [-1,2,8,2,-1],
                                [-1,2,2,2,-1],
                                [-1,-1,-1,-1,-1]]) / 8.0
    image = cv.filter2D(image, -1, kernal_shearing)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

#########################################################################
######################Morphological Transformations######################

# def TransformCombobox(event):
#     global image
#     global photo
#     kernel = np.ones((3,3),np.uint8)
#     transform = event.widget.get()
#     if transform == 'Erosion':
#         image = cv.erode(image,kernel,iterations = 1)
#     elif transform == 'Dilation':
#         image = cv.dilate(image,kernel,iterations = 1)
#     elif transform == 'Opening':
#         image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
#     elif transform == 'Closing':
#         image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
#     elif transform == 'Gradient':
#         image = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
#     elif transform == 'Top Hat':
#         image = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
#     elif transform == 'Black Hat':
#         image = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
#     photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)

def TransformComboboxG(event):
    global image
    global photo
    transform = event.widget.get()
    if transform == 'Erosion':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        invert = cv.bitwise_not(binr)
        image = cv.erode(invert,kernel,iterations=1)
    elif transform == 'Dilation':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        invert = cv.bitwise_not(binr)
        image = cv.dilate(invert, kernel, iterations=1)
    elif transform == 'Opening':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        image = cv.morphologyEx(binr, cv.MORPH_OPEN, kernel, iterations=1)
    elif transform == 'Closing':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        image = cv.morphologyEx(binr, cv.MORPH_CLOSE, kernel, iterations=1)
    elif transform == 'Gradient':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        invert = cv.bitwise_not(binr)
        image = cv.morphologyEx(invert, cv.MORPH_GRADIENT ,kernel)
    elif transform == 'Top Hat':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((13, 13), np.uint8)
        image = cv.morphologyEx(binr, cv.MORPH_TOPHAT ,kernel)
    elif transform == 'Black Hat':
        binr = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        invert = cv.bitwise_not(binr)
        image = cv.morphologyEx(invert, cv.MORPH_BLACKHAT ,kernel)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

#########################################################################

def edgeDetectHor():
    global image
    global photo
    kernal = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    image = cv.filter2D(image, -1, kernal)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def edgeDetectVer():
    global image
    global photo
    kernal = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    image = cv.filter2D(image, -1, kernal)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def edgeDetectDiag():
    global image
    global photo
    kernal = np.array([[2, 1, 0],[1, 0, -1],[0, -1, -2]])
    image = cv.filter2D(image, -1, kernal)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def edgeDetectSobel():
    global image
    global photo
    kernal_1 = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernal_2 = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    kernal_3 = np.array([[2, 1, 0],[1, 0, -1],[0, -1, -2]])
    horizontal = cv.filter2D(image, -1 , kernal_1)
    vertical = cv.filter2D(image, -1 , kernal_2)
    dignal = cv.filter2D(image, -1 , kernal_3)
    dst_1 = cv.addWeighted(vertical,1,horizontal,1,0.0)
    image = cv.addWeighted(dst_1,1,dignal,1,0.0)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def getMeanAndStandardDeviation():
    global image
    global mean
    global stanDiv
    global min
    global max
    mean = statistics.mean(image.ravel())
    stanDiv = statistics.stdev(image.ravel())
    min = mean - stanDiv
    max = mean + stanDiv
    print(mean)
    print(stanDiv)
    print(min)
    print(max)
    
    findEmptyCanves(image)

def edgeDetectLaplacian():
    global image
    global photo
    image = cv.Laplacian(image,cv.CV_64F)
    # kernel = np.array([[0, -1, 0], [-1, 4, -1,], [0, -1, 0]])
    # image = cv.filter2D(image, -1, kernel)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def edgeDetectCanny():
    global image
    global photo
    if Max_Canny_Val.get() == '' and Min_Canny_Val.get() =='':
        max = mean + stanDiv
        min = mean + stanDiv 
    else:
        max = int(Max_Canny_Val.get())
        min = int(Min_Canny_Val.get())
    image = cv.Canny(image, min, max)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def threshold(event):
    global finalEdit
    global image
    global photo
    if Max_threshold_Val.get() == '' and Min_threshold_Val.get() =='':
        max = mean + stanDiv
        min = mean + stanDiv 
    else:
        max = int(Max_threshold_Val.get())
        min = int(Min_threshold_Val.get())
    transform = event.widget.get()
    if transform == 'thresh':
        ret, image = cv.threshold(image, min, max, 0)
    elif transform == 'Binary':
        ret, image = cv.threshold(image, min, max, cv.THRESH_BINARY)
    elif transform == 'Inverse Binary':
        ret, image = cv.threshold(image, min, max, cv.THRESH_BINARY_INV)
    elif transform == 'Truncated':
        ret, image = cv.threshold(image, min, max, cv.THRESH_TRUNC)
    elif transform == 'To-Zero':
        ret, image = cv.threshold(image, min, max, cv.THRESH_TOZERO)
    elif transform == 'Inverse To-Zero':
        ret, image = cv.threshold(image, min, max, cv.THRESH_TOZERO_INV)
    elif transform == 'Otsu':
        ret, image = cv.threshold(image, min, max, cv.THRESH_OTSU)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

#############################################################################################
#contour

# def contour():
#     global image
#     global photo
#     if Max_contour_Val_text.get() == '' and Min_contour_Val_text.get() =='':
#         max = mean + stanDiv
#         min = mean + stanDiv 
#     else:
#         max = int(Max_contour_Val_text.get())
#         min = int(Min_contour_Val_text.get())
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#     edged = cv.Canny(gray, min, max)
#     ret1, edged1 = cv.threshold(image, mean - stanDiv, mean + stanDiv, cv.THRESH_BINARY)
#     plt.imshow(edged1)
#     contours, hierarchy = cv.findContours(edged1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     plt.imshow(edged1)
#     print("Number of Contours found = " + str(len(contours)))
#     cv.drawContours(image, contours, -1, (0,255,0), 3)
#     plt.imshow(image)
#     photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)
#     findEmptyCanves(image)

# def ImageSegmentationContours():
def contour():
    global image
    global photo
    ret1, thresh1 = cv.threshold(image, mean - stanDiv, mean + stanDiv, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours) - 1))
    cv.drawContours(image, contours, -1,color=(0,255,0),thickness=3)
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)


#anwa3 al contours

# def featuresOfContours(event):
#     global image
#     global photo
#     transform = event.widget.get()
#     ret,thresh = cv.threshold(image,127,255,0)
#     im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
#     contours = contours[0]
#     if transform == 'Moments':
#         M = cv.moments(contours)
#         print( M )
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#     elif transform == 'Area':
#         image = cv.contourArea(contours)
#     elif transform == 'Perimeter':
#         image = cv.arcLength(contours,True)
#     elif transform == 'Approximation':
#         epsilon = 0.1*cv.arcLength(contours,True)
#         image = cv.approxPolyDP(contours,epsilon,True)
#     elif transform == 'Hull':
#         image = cv.convexHull(contours)
#     elif transform == 'Convexity':
#         image = cv.isContourConvex(contours)
#     elif transform == 'Straight Rectangle':
#         x,y,w,h = cv.boundingRect(contours)
#         image = cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#     elif transform == 'Rotated Rectangle':
#         rect = cv.minAreaRect(contours)
#         box = cv.boxPoints(rect)
#         box = np.int0(box)
#         image = cv.drawContours(image,[box],0,(0,0,255),2)
#     elif transform == 'Minimum Enclosing Circle':
#         (x,y),radius = cv.minEnclosingCircle(contours)
#         center = (int(x),int(y))
#         radius = int(radius)
#         image = cv.circle(image,center,radius,(0,255,0),2)
#     elif transform == 'Fitting an Ellipse':
#         ellipse = cv.fitEllipse(contours)
#         image = cv.ellipse(image,ellipse,(0,255,0),2)
#     elif transform == 'Fitting a Line':
#         rows,cols = image.shape[:2]
#         [vx,vy,x,y] = cv.fitLine(contours, cv.DIST_L2,0,0.01,0.01)
#         lefty = int((-x*vy/vx) + y)
#         righty = int(((cols-x)*vy/vx)+y)
#         image = cv.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
#     photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)

#############################################################################################

def SegmentationK_means():
    global image
    global photo
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1,3))
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = int(K_Val.get())
    attempts=10
    ret,labels,center=cv.kmeans(pixel_values,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    labels = labels.flatten()
    res = center[labels.flatten()]
    result_image = res.reshape((image.shape))
    image = result_image
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def DetectSimpleGeometricShapes():
    global image
    global photo
    mean = statistics.mean(image.ravel())
    # _, thrash = cv.threshold(image, mean , mean, cv.THRESH_BINARY)
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    for contour in contours:
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        cv.drawContours(image, [approx], 0, (0), 3)
        x, y = approx[0][0]
        if len(approx) == 3:
            cv.putText(image, "Triangle", (x - 25 , y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 4:
            x1 ,y1, w, h = cv.boundingRect(approx)
            aspectRatio = float(w)/h
            print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                cv.putText(image, "Square", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
            else:
                cv.putText(image, "Rectangle", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 5:
            cv.putText(image, "Pentagon", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 6:
            cv.putText(image, "Hexagon", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 7:
            cv.putText(image, "Heptagon", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 8:
            cv.putText(image, "Octagons", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 9:
            cv.putText(image, "Nonagon", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 10:
            cv.putText(image, "Star", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) == 11:
            cv.putText(image, "Undecagon", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        elif len(approx) >= 12 and len(approx) <= 15:
            cv.putText(image, "Elipse", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
        else:
            cv.putText(image, "Circle", (x - 25, y + 25), cv.FONT_HERSHEY_COMPLEX, 1, (155, 0, 155))
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)


################################################################################################

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def Low_pass_filter():
    global image
    global photo
    d0 = int(d0Scale.get())
    plt.figure(figsize=(25, 5), constrained_layout=False)
    plt.subplot(161), plt.imshow(image, "gray"), plt.title("Original Image")
    original = np.fft.fft2(image)
    plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
    center = np.fft.fftshift(original)
    plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")
    LowPassCenter = center * gaussianLP(d0,image.shape)
    plt.subplot(164), plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply \n Low Pass Filter")
    LowPass = np.fft.ifftshift(LowPassCenter)
    plt.subplot(165), plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")
    inverse_LowPass = np.fft.ifft2(LowPass)
    plt.subplot(166), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")
    plt.suptitle("D0:"+str(d0),fontweight="bold")
    plt.subplots_adjust(top=1.0)
    plt.show()
    
    image = np.fft.fft2(image)
    shiftimage = np.fft.fftshift(image)
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    n = 3
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= d0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    gshift = shiftimage * H
    G = np.fft.ifftshift(gshift)
    image = np.abs(np.fft.ifft2(G))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def high_pass_filter():
    global image
    global photo
    d0 = int(d0Scale.get())
    plt.figure(figsize=(25, 5), constrained_layout=False)
    plt.subplot(161), plt.imshow(image, "gray"), plt.title("Original")
    original = np.fft.fft2(image)
    plt.subplot(162), plt.imshow(np.log(1+np.abs(original)), "gray"), plt.title("Spectrum")
    center = np.fft.fftshift(original)
    plt.subplot(163), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Centered Spectrum")
    HighPassCenter = center * gaussianHP(d0,image.shape)
    plt.subplot(164), plt.imshow(np.log(1+np.abs(HighPassCenter)), "gray"), plt.title("Centered Spectrum multiply \n High Pass Filter")
    HighPass = np.fft.ifftshift(HighPassCenter)
    plt.subplot(165), plt.imshow(np.log(1+np.abs(HighPass)), "gray"), plt.title("Decentralize")
    inverse_HighPass = np.fft.ifft2(HighPass)
    plt.subplot(166), plt.imshow(np.abs(inverse_HighPass), "gray"), plt.title("Processed Image")
    plt.suptitle("D0:"+str(d0),fontweight="bold")
    plt.subplots_adjust(top=1.1)
    plt.show()
    
    
    image = np.fft.fft2(image)
    shiftimage = np.fft.fftshift(image)
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    n = 3
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= d0:
                H[u, v] = 0
            else:
                H[u, v] = 1
    gshift = shiftimage * H
    G = np.fft.ifftshift(gshift)
    image = np.abs(np.fft.ifft2(G))
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

def car_plate():
    global image
    global photo
    global screenCnt

    cnts,new = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True) [:30]
    for c in cnts:
        arc = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.015 * arc, True)
        if len(approx) == 4: 
            screenCnt = approx
            break
    x,y,w,h = cv.boundingRect(screenCnt) 
    new_img=image[y:y+h,x:x+w]
    reader = Reader(['en','ar'], gpu=False,verbose=False)
    detection = reader.readtext(new_img)
    # print(detection)
    
    if len(detection) == 0:
        print('Cant Read Number')
        # cv.imwrite("plate_Image.png",image)
        results=arabicocr.arabic_ocr("./plate_Image.png","./plate_Image.png")
        # print(results)
        words=[]
        for i in range(len(results)):	
            word=results[i][1]
            words.append(word)
            print("Car plate number is " + str(words))
        # with open ('./file.txt','w',encoding='utf-8')as myfile:
        #     myfile.write(str(words))
        # messagebox.showinfo(title="Car plate number", message=words)
    else:
        words=[]
        for i in range(len(detection)):	
            word=detection[i][1]
            words.append(word)
            print("Car plate number is " + str(words))
        # with open ('./file.txt','w',encoding='utf-8')as myfile:
        #     myfile.write(str(words))
        # messagebox.showinfo(title="Car plate number", message=words)
    
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor=NW)
    findEmptyCanves(image)

# def car_plate():
#     global image
#     global photo
    
#     image = imutils.resize(image, width=300 )
#     gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     gray_image = cv.bilateralFilter(gray_image, 11, 17, 17) 
#     edged = cv.Canny(gray_image, 30, 200) 
#     cnts,new = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#     image1=image.copy()
#     cv.drawContours(image1,cnts,-1,(0,255,0),3)
#     cnts = sorted(cnts, key = cv.contourArea, reverse = True) [:30]
#     screenCnt = None
#     image2 = image.copy()
#     cv.drawContours(image2,cnts,-1,(0,255,0),3)
#     i=7
#     for c in cnts:
#         perimeter = cv.arcLength(c, True)
#         approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
#         if len(approx) == 4: 
#             screenCnt = approx
#             x,y,w,h = cv.boundingRect(c) 
#             new_img=image[y:y+h,x:x+w]
#             cv.imwrite('./'+str(i)+'.png',new_img)
#             i+=1
#             break
#     cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
#     Cropped_loc = image
#     plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
#     print("Number plate is:", plate)
#     image =Cropped_loc 
#     photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)
#     findEmptyCanves(image)

def exit():
    pro.destroy()

# def get_x_and_y(event):
#     global lasx , lasy
#     lasx , lasy = event.x , event.y

# def draw_smth(event):
#     global lasx , lasy
#     canvas.create_line((lasx , lasy, event.x, event.y), fill = 'red',width =2)
#     lasx , lasy = event.x , event.y

# def select_area():
#     canvas.bind("<Button-1>", get_x_and_y)
#     canvas.bind("<B1-Motion>", draw_smth)

# mask = np.ones((800, 600))
# if lasx < 600 and lasx >= 0 and lasy < 800 and lasy >= 0:
#     mask[lasx][lasy] = 0
#     mask[lasx+1][lasy+1] = 0
#     mask[lasx-1][lasy-1] = 0
#     mask[lasx+1][lasy-1] = 0
#     mask[lasx-1][lasy+1] = 0

# def retrun_shape(image_in):
#     global image
#     global photo
#     image = image_in
#     gray = image_in
#     edged = cv.Canny (gray, 30, 200)
#     contours, hierarchy = cv.findContours (edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     cv.drawContours (image, contours, -1, (0, 0, 0), 3)
#     th, im_th = cv.threshold (image, 200, 255, cv.THRESH_BINARY_INV)
#     im_floodfill = im_th.copy()
#     h, w = im_th.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#     cv.floodFill(im_floodfill, mask, (0,0), (255,255,255))
#     image = im_floodfill
#     photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
#     canvas.create_image(0, 0, image=photo, anchor=NW)
#     findEmptyCanves(image)

# def show_mask():
#     global image_for_mask_multiplication 
#     mask_3_channels = np.ones((490, 500, 3))
#     image_mattt = (mask * 255).astype(np.uint8) 
#     the_real_mask = retrun_shape(image_mattt) 
#     mask_3_channels[:,:,0] = the_real_mask/255 
#     mask_3_channels[:,:,1] = the_real_mask/255 
#     mask_3_channels[:,:,2] = the_real_mask/255
#     real_area = np.array(image_for_mask_multiplication) * mask_3_channels 
#     real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
#     real_area.show()

def thm1():
#light_mood

    iconPath = r"./Light_Icons/"
    Icon_color = 'white'
    Text_color_bg = 'white'
    Text_Color_font = 'black'
    entity_text_Color = 'white'
    main_color_bg = 'white'
    fr_up_color_bg = 'white'
    fr_down_color_bg = 'white'
    fr_left_1_color_bg = 'white'
    fr_left_2_color_bg = 'white'
    fr_left_3_color_bg = "white"
    fr2_right_1_color_bg = 'white'
    fr2_right_2_color_bg = 'white'
    fr2_right_3_color_bg = 'white'

    fr_up.config(bg=fr_down_color_bg)
    fr_down.config(bg=fr_up_color_bg)
    fr_left_1.config(bg=fr_left_1_color_bg)
    fr_left_2.config(bg=fr_left_2_color_bg)
    fr_left_3.config(bg=fr_left_3_color_bg)
    fr2_right_1.config(bg=fr2_right_1_color_bg)
    fr2_right_2.config(bg=fr2_right_2_color_bg)
    fr2_right_3.config(bg=fr2_right_3_color_bg)
    canvas.config(bg=main_color_bg)
    pro.config(background=main_color_bg)
    canvasTest_1.config(bg=fr2_right_1_color_bg)
    canvasTest_2.config(bg=fr2_right_1_color_bg)

    browse_icon.config(file= iconPath + r"Browse.png")
    browseBtn.config(image = browse_icon, bg=Icon_color)
    save_icon.config(file= iconPath + r"save.png")
    saveBtn.config(image = save_icon, bg=Icon_color)
    saveAs_icon.config(file= iconPath + r"save as.png")
    saveAsBtn.config(image = saveAs_icon, bg=Icon_color)
    reset_icon.config(file= iconPath + r"reset.png")
    resetBtn.config(image = reset_icon, bg=Icon_color)
    restore_icon.config(file= iconPath + r"restore.png")
    restoreBtn.config(image = restore_icon, bg=Icon_color)
    gray_scale_icon.config(file= iconPath + r"grayscale.png")
    gray_scaleBtn.config(image = gray_scale_icon, bg=Icon_color)
    rotateRIcon.config(file= iconPath + r"rotate clockwise.png")
    rotateRBtn.config(image = rotateRIcon, bg=Icon_color)
    rotateLIcon.config(file= iconPath + r"rotate anticlockwise.png")
    rotateLBtn.config(image = rotateLIcon, bg=Icon_color)
    Move_up.config(file= iconPath + r"arrow up.png")
    transformupBtn.config(image = Move_up, bg=Icon_color)
    Move_left.config(file= iconPath + r"arrow right.png")
    transformleftBtn.config(image = Move_left, bg=Icon_color)
    Move_right.config(file= iconPath + r"arrow left.png")
    transformrightBtn.config(image = Move_right, bg=Icon_color)
    Move_down.config(file= iconPath + r"arrow down.png")
    transformdownBtn.config(image = Move_down, bg=Icon_color)
    Move_up_left.config(file= iconPath + r"arrow down right.png")
    transformulBtn.config(image = Move_up_left, bg=Icon_color)
    Move_up_right.config(file= iconPath + r"arrow down left.png")
    transformurBtn.config(image = Move_up_right, bg=Icon_color)
    Move_down_left.config(file= iconPath + r"arrow up right.png")
    transformdlBtn.config(image = Move_down_left, bg=Icon_color)
    Move_down_right.config(file= iconPath + r"arrow up left.png")
    transformdrBtn.config(image = Move_down_right, bg=Icon_color)
    skewing_R_Transformation.config(file= iconPath + r"skewing right.png")
    skewingRTransformationBtn.config(image = skewing_R_Transformation, bg=Icon_color)
    skewing_L_Transformation.config(file= iconPath + r"skewing left.png")
    skewingLTransformationBtn.config(image = skewing_L_Transformation, bg=Icon_color)
    flip_H.config(file= iconPath + r"flip horizontal.png")
    fliphBtn.config(image = flip_H, bg=Icon_color)
    flip_V.config(file= iconPath + r"flip vertical.png")
    flipvBtn.config(image = flip_V, bg=Icon_color)
    crop_icon.config(file= iconPath + r"crop.png")
    cropBtn.config(image = crop_icon, bg=Icon_color)
    browse_merge_icon.config(file= iconPath + r"Browse.png")
    browse_mergeBtn.config(image = browse_merge_icon, bg=Icon_color)
    merge_icon.config(file= iconPath + r"merge.png")
    mergeBtn.config(image = merge_icon, bg=Icon_color)
    histGraph_icon.config(file= iconPath + r"graph.png")
    histGraphBtn.config(image = histGraph_icon, bg=Icon_color)
    equalHist_icon.config(file= iconPath + r"equalHist.png")
    equalHistBtn.config(image = equalHist_icon, bg=Icon_color)
    negative_icon.config(file= iconPath + r"Negative.png")
    negativeBtn.config(image = negative_icon, bg=Icon_color)
    LogarithmicTrans_icon.config(file= iconPath + r"Logarithmic.png")
    LogarithmicTransBtn.config(image = LogarithmicTrans_icon, bg=Icon_color)
    PowerTrans_icon.config(file= iconPath + r"power.png")
    PowerTransBtn.config(image = PowerTrans_icon, bg=Icon_color)
    Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Bit_PlaneBtn.config(image = Bit_Plane_icon, bg=Icon_color)
    Last_Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Last_Bit_PlaneBtn.config(image = Last_Bit_Plane_icon, bg=Icon_color)
    Gray_level_icon.config(file= iconPath + r"gray level.png")
    Gray_levelBtn.config(image = Gray_level_icon, bg=Icon_color)
    sharping1_icon.config(file= iconPath + r"sharpen1.png")
    sharpingBtn1.config(image = sharping1_icon, bg=Icon_color)
    sharping2_icon.config(file= iconPath + r"sharpen2.png")
    sharpingBtn2.config(image = sharping2_icon, bg=Icon_color)
    sharping3_icon.config(file= iconPath + r"sharpen3.png")
    sharpingBtn3.config(image = sharping3_icon, bg=Icon_color)
    edgeBtn_Hor_icon.config(file= iconPath + r"horizontal edge.png")
    edgeBtn_Hor.config(image = edgeBtn_Hor_icon, bg=Icon_color)
    edgeBtn_Ver_icon.config(file= iconPath + r"vertical edge.png")
    edgeBtn_Ver.config(image = edgeBtn_Ver_icon, bg=Icon_color)
    edgeBtn_Diag_icon.config(file= iconPath + r"dignal edge.png")
    edgeBtn_Diag.config(image = edgeBtn_Diag_icon, bg=Icon_color)
    edgeBtn_Sobel_icon.config(file= iconPath + r"sobel.png")
    edgeBtn_Sobel.config(image = edgeBtn_Sobel_icon, bg=Icon_color)
    edgeBtn_laplacian_icon.config(file= iconPath + r"laplacian.png")
    edgeBtn_laplacian.config(image = edgeBtn_laplacian_icon, bg=Icon_color)
    edgeBtn_Canny_icon.config(file= iconPath + r"canny.png")
    edgeBtn_Canny.config(image = edgeBtn_Canny_icon, bg=Icon_color)
    contour_icon.config(file= iconPath + r"contour.png")
    contourBtn.config(image = contour_icon, bg=Icon_color)
    SegmentationKmeans_icon.config(file= iconPath + r"k-means.png")
    SegmentationKmeansBtn.config(image = SegmentationKmeans_icon, bg=Icon_color)
    DetectSimpleGeometricShapes_icon.config(file= iconPath + r"Geometric Shapes.png")
    DetectSimpleGeometricShapesBtn.config(image = DetectSimpleGeometricShapes_icon, bg=Icon_color)
    car_plate_icon.config(file= iconPath + r"car plate.png")
    car_plateBtn.config(image = car_plate_icon, bg=Icon_color)
    Low_pass_filter_icon.config(file= iconPath + r"los pass.png")
    Low_pass_filterBtn.config(image = Low_pass_filter_icon, bg=Icon_color)
    high_pass_filter_icon.config(file= iconPath + r"high pass.png")
    high_pass_filterBtn.config(image = high_pass_filter_icon, bg=Icon_color)
    Lossless_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossless_compression_SaveAsBtn.config(image = Lossless_compression_SaveAs_icon, bg=Icon_color)
    Lossy_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossy_compression_SaveAsBtn.config(image = Lossy_compression_SaveAs_icon, bg=Icon_color)
    exit_icon.config(file= iconPath + r"exit.png")
    exitBtn.config(image = exit_icon, bg=Icon_color)
    Light_mood_icon.config(file= iconPath + r"light.png")
    Light_moodBtn.config(image = Light_mood_icon, bg=Icon_color)
    Dark_mood_icon.config(file= iconPath + r"dark.png")
    Dark_moodBtn.config(image = Dark_mood_icon, bg=Icon_color)
    Gray_mood_icon.config(file= iconPath + r"gray.png")
    Gray_moodBtn.config(image = Gray_mood_icon, bg=Icon_color)
    Dark_Gray_mood_icon.config(file= iconPath + r"dark gray.png")
    Dark_Gray_moodBtn.config(image = Dark_Gray_mood_icon, bg=Icon_color)

    browse_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    save_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    saveAs_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    reset_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    restore_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    gray_scale_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    rotate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    transform_text.config(fg=Text_Color_font, bg=Text_color_bg)
    skewing_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    flip_text.config(fg=Text_Color_font,bg=Text_color_bg)
    crop_text.config(fg=Text_Color_font, bg=Text_color_bg)
    browse_merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    histGraph_text.config(fg=Text_Color_font, bg=Text_color_bg)
    equalHist_text.config(fg=Text_Color_font, bg=Text_color_bg)
    negative_text.config(fg=Text_Color_font, bg=Text_color_bg)
    LogarithmicTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    PowerTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Last_Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Gray_level_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping1_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping3_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Hor_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Ver_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Diag_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Sobel_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_laplacian_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Canny_text.config(fg=Text_Color_font, bg=Text_color_bg)
    contour_text.config(fg=Text_Color_font, bg=Text_color_bg)
    SegmentationKmeans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    DetectSimpleGeometricShapes_text.config(fg=Text_Color_font, bg=Text_color_bg)
    car_plate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Low_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    high_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossless_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossy_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    themes_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    exit_text.config(fg=Text_Color_font,  bg=Text_color_bg )

    gamma_text.config(fg=Text_Color_font, bg=Text_color_bg )
    gammaVal.config(fg=Text_Color_font,bg=entity_text_Color)
    Bit_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    blur_text.config(fg=Text_Color_font, bg=Text_color_bg)
    MorphologicalG_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    threshold_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    K_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    K_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    d0Scale_text.config(fg=Text_Color_font, bg=Text_color_bg)
    d0Scale.config(fg=Text_Color_font , bg= Text_color_bg)
    # quality_val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val.config(fg=Text_Color_font,bg=entity_text_Color)
    # quality_val2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val2.config(fg=Text_Color_font,bg=entity_text_Color)

    pro.update()

def thm2():
#Dark_mood

    iconPath = r"./Dark_Icons/"
    Icon_color = 'black'
    Text_color_bg = 'black'
    Text_Color_font = 'white'
    entity_text_Color = 'black'
    main_color_bg = 'black'
    fr_up_color_bg = 'black'
    fr_down_color_bg = 'black'
    fr_left_1_color_bg = 'black'
    fr_left_2_color_bg = 'black'
    fr_left_3_color_bg = "black"
    fr2_right_1_color_bg = 'black'
    fr2_right_2_color_bg = 'black'
    fr2_right_3_color_bg = 'black'

    fr_up.config(bg=fr_down_color_bg)
    fr_down.config(bg=fr_up_color_bg)
    fr_left_1.config(bg=fr_left_1_color_bg)
    fr_left_2.config(bg=fr_left_2_color_bg)
    fr_left_3.config(bg=fr_left_3_color_bg)
    fr2_right_1.config(bg=fr2_right_1_color_bg)
    fr2_right_2.config(bg=fr2_right_2_color_bg)
    fr2_right_3.config(bg=fr2_right_3_color_bg)
    canvas.config(bg=main_color_bg)
    pro.config(background=main_color_bg)
    canvasTest_1.config(bg=fr2_right_1_color_bg)
    canvasTest_2.config(bg=fr2_right_1_color_bg)

    browse_icon.config(file= iconPath + r"Browse.png")
    browseBtn.config(image = browse_icon, bg=Icon_color)
    save_icon.config(file= iconPath + r"save.png")
    saveBtn.config(image = save_icon, bg=Icon_color)
    saveAs_icon.config(file= iconPath + r"save as.png")
    saveAsBtn.config(image = saveAs_icon, bg=Icon_color)
    reset_icon.config(file= iconPath + r"reset.png")
    resetBtn.config(image = reset_icon, bg=Icon_color)
    restore_icon.config(file= iconPath + r"restore.png")
    restoreBtn.config(image = restore_icon, bg=Icon_color)
    gray_scale_icon.config(file= iconPath + r"grayscale.png")
    gray_scaleBtn.config(image = gray_scale_icon, bg=Icon_color)
    rotateRIcon.config(file= iconPath + r"rotate clockwise.png")
    rotateRBtn.config(image = rotateRIcon, bg=Icon_color)
    rotateLIcon.config(file= iconPath + r"rotate anticlockwise.png")
    rotateLBtn.config(image = rotateLIcon, bg=Icon_color)
    Move_up.config(file= iconPath + r"arrow up.png")
    transformupBtn.config(image = Move_up, bg=Icon_color)
    Move_left.config(file= iconPath + r"arrow right.png")
    transformleftBtn.config(image = Move_left, bg=Icon_color)
    Move_right.config(file= iconPath + r"arrow left.png")
    transformrightBtn.config(image = Move_right, bg=Icon_color)
    Move_down.config(file= iconPath + r"arrow down.png")
    transformdownBtn.config(image = Move_down, bg=Icon_color)
    Move_up_left.config(file= iconPath + r"arrow down right.png")
    transformulBtn.config(image = Move_up_left, bg=Icon_color)
    Move_up_right.config(file= iconPath + r"arrow down left.png")
    transformurBtn.config(image = Move_up_right, bg=Icon_color)
    Move_down_left.config(file= iconPath + r"arrow up right.png")
    transformdlBtn.config(image = Move_down_left, bg=Icon_color)
    Move_down_right.config(file= iconPath + r"arrow up left.png")
    transformdrBtn.config(image = Move_down_right, bg=Icon_color)
    skewing_R_Transformation.config(file= iconPath + r"skewing right.png")
    skewingRTransformationBtn.config(image = skewing_R_Transformation, bg=Icon_color)
    skewing_L_Transformation.config(file= iconPath + r"skewing left.png")
    skewingLTransformationBtn.config(image = skewing_L_Transformation, bg=Icon_color)
    flip_H.config(file= iconPath + r"flip horizontal.png")
    fliphBtn.config(image = flip_H, bg=Icon_color)
    flip_V.config(file= iconPath + r"flip vertical.png")
    flipvBtn.config(image = flip_V, bg=Icon_color)
    crop_icon.config(file= iconPath + r"crop.png")
    cropBtn.config(image = crop_icon, bg=Icon_color)
    browse_merge_icon.config(file= iconPath + r"Browse.png")
    browse_mergeBtn.config(image = browse_merge_icon, bg=Icon_color)
    merge_icon.config(file= iconPath + r"merge.png")
    mergeBtn.config(image = merge_icon, bg=Icon_color)
    histGraph_icon.config(file= iconPath + r"graph.png")
    histGraphBtn.config(image = histGraph_icon, bg=Icon_color)
    equalHist_icon.config(file= iconPath + r"equalHist.png")
    equalHistBtn.config(image = equalHist_icon, bg=Icon_color)
    negative_icon.config(file= iconPath + r"Negative.png")
    negativeBtn.config(image = negative_icon, bg=Icon_color)
    LogarithmicTrans_icon.config(file= iconPath + r"Logarithmic.png")
    LogarithmicTransBtn.config(image = LogarithmicTrans_icon, bg=Icon_color)
    PowerTrans_icon.config(file= iconPath + r"power.png")
    PowerTransBtn.config(image = PowerTrans_icon, bg=Icon_color)
    Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Bit_PlaneBtn.config(image = Bit_Plane_icon, bg=Icon_color)
    Last_Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Last_Bit_PlaneBtn.config(image = Last_Bit_Plane_icon, bg=Icon_color)
    Gray_level_icon.config(file= iconPath + r"gray level.png")
    Gray_levelBtn.config(image = Gray_level_icon, bg=Icon_color)
    sharping1_icon.config(file= iconPath + r"sharpen1.png")
    sharpingBtn1.config(image = sharping1_icon, bg=Icon_color)
    sharping2_icon.config(file= iconPath + r"sharpen2.png")
    sharpingBtn2.config(image = sharping2_icon, bg=Icon_color)
    sharping3_icon.config(file= iconPath + r"sharpen3.png")
    sharpingBtn3.config(image = sharping3_icon, bg=Icon_color)
    edgeBtn_Hor_icon.config(file= iconPath + r"horizontal edge.png")
    edgeBtn_Hor.config(image = edgeBtn_Hor_icon, bg=Icon_color)
    edgeBtn_Ver_icon.config(file= iconPath + r"vertical edge.png")
    edgeBtn_Ver.config(image = edgeBtn_Ver_icon, bg=Icon_color)
    edgeBtn_Diag_icon.config(file= iconPath + r"dignal edge.png")
    edgeBtn_Diag.config(image = edgeBtn_Diag_icon, bg=Icon_color)
    edgeBtn_Sobel_icon.config(file= iconPath + r"sobel.png")
    edgeBtn_Sobel.config(image = edgeBtn_Sobel_icon, bg=Icon_color)
    edgeBtn_laplacian_icon.config(file= iconPath + r"laplacian.png")
    edgeBtn_laplacian.config(image = edgeBtn_laplacian_icon, bg=Icon_color)
    edgeBtn_Canny_icon.config(file= iconPath + r"canny.png")
    edgeBtn_Canny.config(image = edgeBtn_Canny_icon, bg=Icon_color)
    contour_icon.config(file= iconPath + r"contour.png")
    contourBtn.config(image = contour_icon, bg=Icon_color)
    SegmentationKmeans_icon.config(file= iconPath + r"k-means.png")
    SegmentationKmeansBtn.config(image = SegmentationKmeans_icon, bg=Icon_color)
    DetectSimpleGeometricShapes_icon.config(file= iconPath + r"Geometric Shapes.png")
    DetectSimpleGeometricShapesBtn.config(image = DetectSimpleGeometricShapes_icon, bg=Icon_color)
    car_plate_icon.config(file= iconPath + r"car plate.png")
    car_plateBtn.config(image = car_plate_icon, bg=Icon_color)
    Low_pass_filter_icon.config(file= iconPath + r"los pass.png")
    Low_pass_filterBtn.config(image = Low_pass_filter_icon, bg=Icon_color)
    high_pass_filter_icon.config(file= iconPath + r"high pass.png")
    high_pass_filterBtn.config(image = high_pass_filter_icon, bg=Icon_color)
    Lossless_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossless_compression_SaveAsBtn.config(image = Lossless_compression_SaveAs_icon, bg=Icon_color)
    Lossy_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossy_compression_SaveAsBtn.config(image = Lossy_compression_SaveAs_icon, bg=Icon_color)
    exit_icon.config(file= iconPath + r"exit.png")
    exitBtn.config(image = exit_icon, bg=Icon_color)
    Light_mood_icon.config(file= iconPath + r"light.png")
    Light_moodBtn.config(image = Light_mood_icon, bg=Icon_color)
    Dark_mood_icon.config(file= iconPath + r"dark.png")
    Dark_moodBtn.config(image = Dark_mood_icon, bg=Icon_color)
    Gray_mood_icon.config(file= iconPath + r"gray.png")
    Gray_moodBtn.config(image = Gray_mood_icon, bg='white')
    Dark_Gray_mood_icon.config(file= iconPath + r"dark gray.png")
    Dark_Gray_moodBtn.config(image = Dark_Gray_mood_icon, bg='white')

    browse_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    save_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    saveAs_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    reset_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    restore_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    gray_scale_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    rotate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    transform_text.config(fg=Text_Color_font, bg=Text_color_bg)
    skewing_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    flip_text.config(fg=Text_Color_font,bg=Text_color_bg)
    crop_text.config(fg=Text_Color_font, bg=Text_color_bg)
    browse_merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    histGraph_text.config(fg=Text_Color_font, bg=Text_color_bg)
    equalHist_text.config(fg=Text_Color_font, bg=Text_color_bg)
    negative_text.config(fg=Text_Color_font, bg=Text_color_bg)
    LogarithmicTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    PowerTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Last_Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Gray_level_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping1_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping3_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Hor_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Ver_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Diag_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Sobel_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_laplacian_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Canny_text.config(fg=Text_Color_font, bg=Text_color_bg)
    contour_text.config(fg=Text_Color_font, bg=Text_color_bg)
    SegmentationKmeans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    DetectSimpleGeometricShapes_text.config(fg=Text_Color_font, bg=Text_color_bg)
    car_plate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Low_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    high_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossless_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossy_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    themes_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    exit_text.config(fg=Text_Color_font,  bg=Text_color_bg )

    gamma_text.config(fg=Text_Color_font, bg=Text_color_bg )
    gammaVal.config(fg=Text_Color_font,bg=entity_text_Color)
    Bit_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    blur_text.config(fg=Text_Color_font, bg=Text_color_bg)
    MorphologicalG_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    threshold_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    K_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    K_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    d0Scale_text.config(fg=Text_Color_font, bg=Text_color_bg)
    d0Scale.config(fg=Text_Color_font , bg= Text_color_bg)
    # quality_val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val.config(fg=Text_Color_font,bg=entity_text_Color)
    # quality_val2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val2.config(fg=Text_Color_font,bg=entity_text_Color)

    pro.update()

def thm3():
#Dark_Gray_mood    

    iconPath = r"./Dark_Icons/"
    Icon_color = 'black'
    Text_color_bg = '#212529'
    Text_Color_font = 'white'
    entity_text_Color = '#333533'
    main_color_bg = '#333533'
    fr_up_color_bg = 'gray20'
    fr_down_color_bg = 'gray20'
    fr_left_1_color_bg = '#252422'
    fr_left_2_color_bg = '#343a40'
    fr_left_3_color_bg = "#252422"
    fr2_right_1_color_bg = '#343a40'
    fr2_right_2_color_bg = '#252422'
    fr2_right_3_color_bg = '#343a40'

    fr_up.config(bg=fr_down_color_bg)
    fr_down.config(bg=fr_up_color_bg)
    fr_left_1.config(bg=fr_left_1_color_bg)
    fr_left_2.config(bg=fr_left_2_color_bg)
    fr_left_3.config(bg=fr_left_3_color_bg)
    fr2_right_1.config(bg=fr2_right_1_color_bg)
    fr2_right_2.config(bg=fr2_right_2_color_bg)
    fr2_right_3.config(bg=fr2_right_3_color_bg)
    canvas.config(bg=main_color_bg)
    pro.config(background=main_color_bg)
    canvasTest_1.config(bg=fr2_right_1_color_bg)
    canvasTest_2.config(bg=fr2_right_1_color_bg)

    browse_icon.config(file= iconPath + r"Browse.png")
    browseBtn.config(image = browse_icon, bg=Icon_color)
    save_icon.config(file= iconPath + r"save.png")
    saveBtn.config(image = save_icon, bg=Icon_color)
    saveAs_icon.config(file= iconPath + r"save as.png")
    saveAsBtn.config(image = saveAs_icon, bg=Icon_color)
    reset_icon.config(file= iconPath + r"reset.png")
    resetBtn.config(image = reset_icon, bg=Icon_color)
    restore_icon.config(file= iconPath + r"restore.png")
    restoreBtn.config(image = restore_icon, bg=Icon_color)
    gray_scale_icon.config(file= iconPath + r"grayscale.png")
    gray_scaleBtn.config(image = gray_scale_icon, bg=Icon_color)
    rotateRIcon.config(file= iconPath + r"rotate clockwise.png")
    rotateRBtn.config(image = rotateRIcon, bg=Icon_color)
    rotateLIcon.config(file= iconPath + r"rotate anticlockwise.png")
    rotateLBtn.config(image = rotateLIcon, bg=Icon_color)
    Move_up.config(file= iconPath + r"arrow up.png")
    transformupBtn.config(image = Move_up, bg=Icon_color)
    Move_left.config(file= iconPath + r"arrow right.png")
    transformleftBtn.config(image = Move_left, bg=Icon_color)
    Move_right.config(file= iconPath + r"arrow left.png")
    transformrightBtn.config(image = Move_right, bg=Icon_color)
    Move_down.config(file= iconPath + r"arrow down.png")
    transformdownBtn.config(image = Move_down, bg=Icon_color)
    Move_up_left.config(file= iconPath + r"arrow down right.png")
    transformulBtn.config(image = Move_up_left, bg=Icon_color)
    Move_up_right.config(file= iconPath + r"arrow down left.png")
    transformurBtn.config(image = Move_up_right, bg=Icon_color)
    Move_down_left.config(file= iconPath + r"arrow up right.png")
    transformdlBtn.config(image = Move_down_left, bg=Icon_color)
    Move_down_right.config(file= iconPath + r"arrow up left.png")
    transformdrBtn.config(image = Move_down_right, bg=Icon_color)
    skewing_R_Transformation.config(file= iconPath + r"skewing right.png")
    skewingRTransformationBtn.config(image = skewing_R_Transformation, bg=Icon_color)
    skewing_L_Transformation.config(file= iconPath + r"skewing left.png")
    skewingLTransformationBtn.config(image = skewing_L_Transformation, bg=Icon_color)
    flip_H.config(file= iconPath + r"flip horizontal.png")
    fliphBtn.config(image = flip_H, bg=Icon_color)
    flip_V.config(file= iconPath + r"flip vertical.png")
    flipvBtn.config(image = flip_V, bg=Icon_color)
    crop_icon.config(file= iconPath + r"crop.png")
    cropBtn.config(image = crop_icon, bg=Icon_color)
    browse_merge_icon.config(file= iconPath + r"Browse.png")
    browse_mergeBtn.config(image = browse_merge_icon, bg=Icon_color)
    merge_icon.config(file= iconPath + r"merge.png")
    mergeBtn.config(image = merge_icon, bg=Icon_color)
    histGraph_icon.config(file= iconPath + r"graph.png")
    histGraphBtn.config(image = histGraph_icon, bg=Icon_color)
    equalHist_icon.config(file= iconPath + r"equalHist.png")
    equalHistBtn.config(image = equalHist_icon, bg=Icon_color)
    negative_icon.config(file= iconPath + r"Negative.png")
    negativeBtn.config(image = negative_icon, bg=Icon_color)
    LogarithmicTrans_icon.config(file= iconPath + r"Logarithmic.png")
    LogarithmicTransBtn.config(image = LogarithmicTrans_icon, bg=Icon_color)
    PowerTrans_icon.config(file= iconPath + r"power.png")
    PowerTransBtn.config(image = PowerTrans_icon, bg=Icon_color)
    Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Bit_PlaneBtn.config(image = Bit_Plane_icon, bg=Icon_color)
    Last_Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Last_Bit_PlaneBtn.config(image = Last_Bit_Plane_icon, bg=Icon_color)
    Gray_level_icon.config(file= iconPath + r"gray level.png")
    Gray_levelBtn.config(image = Gray_level_icon, bg=Icon_color)
    sharping1_icon.config(file= iconPath + r"sharpen1.png")
    sharpingBtn1.config(image = sharping1_icon, bg=Icon_color)
    sharping2_icon.config(file= iconPath + r"sharpen2.png")
    sharpingBtn2.config(image = sharping2_icon, bg=Icon_color)
    sharping3_icon.config(file= iconPath + r"sharpen3.png")
    sharpingBtn3.config(image = sharping3_icon, bg=Icon_color)
    edgeBtn_Hor_icon.config(file= iconPath + r"horizontal edge.png")
    edgeBtn_Hor.config(image = edgeBtn_Hor_icon, bg=Icon_color)
    edgeBtn_Ver_icon.config(file= iconPath + r"vertical edge.png")
    edgeBtn_Ver.config(image = edgeBtn_Ver_icon, bg=Icon_color)
    edgeBtn_Diag_icon.config(file= iconPath + r"dignal edge.png")
    edgeBtn_Diag.config(image = edgeBtn_Diag_icon, bg=Icon_color)
    edgeBtn_Sobel_icon.config(file= iconPath + r"sobel.png")
    edgeBtn_Sobel.config(image = edgeBtn_Sobel_icon, bg=Icon_color)
    edgeBtn_laplacian_icon.config(file= iconPath + r"laplacian.png")
    edgeBtn_laplacian.config(image = edgeBtn_laplacian_icon, bg=Icon_color)
    edgeBtn_Canny_icon.config(file= iconPath + r"canny.png")
    edgeBtn_Canny.config(image = edgeBtn_Canny_icon, bg=Icon_color)
    contour_icon.config(file= iconPath + r"contour.png")
    contourBtn.config(image = contour_icon, bg=Icon_color)
    SegmentationKmeans_icon.config(file= iconPath + r"k-means.png")
    SegmentationKmeansBtn.config(image = SegmentationKmeans_icon, bg=Icon_color)
    DetectSimpleGeometricShapes_icon.config(file= iconPath + r"Geometric Shapes.png")
    DetectSimpleGeometricShapesBtn.config(image = DetectSimpleGeometricShapes_icon, bg=Icon_color)
    car_plate_icon.config(file= iconPath + r"car plate.png")
    car_plateBtn.config(image = car_plate_icon, bg=Icon_color)
    Low_pass_filter_icon.config(file= iconPath + r"los pass.png")
    Low_pass_filterBtn.config(image = Low_pass_filter_icon, bg=Icon_color)
    high_pass_filter_icon.config(file= iconPath + r"high pass.png")
    high_pass_filterBtn.config(image = high_pass_filter_icon, bg=Icon_color)
    Lossless_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossless_compression_SaveAsBtn.config(image = Lossless_compression_SaveAs_icon, bg=Icon_color)
    Lossy_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossy_compression_SaveAsBtn.config(image = Lossy_compression_SaveAs_icon, bg=Icon_color)
    exit_icon.config(file= iconPath + r"exit.png")
    exitBtn.config(image = exit_icon, bg=Icon_color)
    Light_mood_icon.config(file= iconPath + r"light.png")
    Light_moodBtn.config(image = Light_mood_icon, bg=Icon_color)
    Dark_mood_icon.config(file= iconPath + r"dark.png")
    Dark_moodBtn.config(image = Dark_mood_icon, bg=Icon_color)
    Gray_mood_icon.config(file= iconPath + r"gray.png")
    Gray_moodBtn.config(image = Gray_mood_icon, bg='white')
    Dark_Gray_mood_icon.config(file= iconPath + r"dark gray.png")
    Dark_Gray_moodBtn.config(image = Dark_Gray_mood_icon, bg='white')

    browse_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    save_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    saveAs_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    reset_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    restore_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    gray_scale_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    rotate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    transform_text.config(fg=Text_Color_font, bg=Text_color_bg)
    skewing_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    flip_text.config(fg=Text_Color_font,bg=Text_color_bg)
    crop_text.config(fg=Text_Color_font, bg=Text_color_bg)
    browse_merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    histGraph_text.config(fg=Text_Color_font, bg=Text_color_bg)
    equalHist_text.config(fg=Text_Color_font, bg=Text_color_bg)
    negative_text.config(fg=Text_Color_font, bg=Text_color_bg)
    LogarithmicTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    PowerTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Last_Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Gray_level_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping1_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping3_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Hor_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Ver_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Diag_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Sobel_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_laplacian_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Canny_text.config(fg=Text_Color_font, bg=Text_color_bg)
    contour_text.config(fg=Text_Color_font, bg=Text_color_bg)
    SegmentationKmeans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    DetectSimpleGeometricShapes_text.config(fg=Text_Color_font, bg=Text_color_bg)
    car_plate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Low_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    high_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossless_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossy_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    themes_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    exit_text.config(fg=Text_Color_font,  bg=Text_color_bg )

    gamma_text.config(fg=Text_Color_font, bg=Text_color_bg )
    gammaVal.config(fg=Text_Color_font,bg=entity_text_Color)
    Bit_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    blur_text.config(fg=Text_Color_font, bg=Text_color_bg)
    MorphologicalG_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    threshold_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    K_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    K_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    d0Scale_text.config(fg=Text_Color_font, bg=Text_color_bg)
    d0Scale.config(fg=Text_Color_font , bg= Text_color_bg)
    # quality_val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val.config(fg=Text_Color_font,bg=entity_text_Color)
    # quality_val2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val2.config(fg=Text_Color_font,bg=entity_text_Color)

    pro.update()

def thm4():
#Gray_mood    

    iconPath = r"./Light_Icons/"
    Icon_color = 'white'
    Text_color_bg = 'gray33'
    Text_Color_font = 'black'
    entity_text_Color = 'gray20'
    main_color_bg = 'gray33'
    fr_up_color_bg = 'gray20'
    fr_down_color_bg = 'gray20'
    fr_left_1_color_bg = 'gray20'
    fr_left_2_color_bg = '#252422'
    fr_left_3_color_bg = 'gray20'
    fr2_right_1_color_bg = "#252422"
    fr2_right_2_color_bg = 'gray20'
    fr2_right_3_color_bg = '#252422'

    fr_up.config(bg=fr_down_color_bg)
    fr_down.config(bg=fr_up_color_bg)
    fr_left_1.config(bg=fr_left_1_color_bg)
    fr_left_2.config(bg=fr_left_2_color_bg)
    fr_left_3.config(bg=fr_left_3_color_bg)
    fr2_right_1.config(bg=fr2_right_1_color_bg)
    fr2_right_2.config(bg=fr2_right_2_color_bg)
    fr2_right_3.config(bg=fr2_right_3_color_bg)
    canvas.config(bg=main_color_bg)
    pro.config(background=main_color_bg)
    canvasTest_1.config(bg=fr2_right_1_color_bg)
    canvasTest_2.config(bg=fr2_right_1_color_bg)

    browse_icon.config(file= iconPath + r"Browse.png")
    browseBtn.config(image = browse_icon, bg=Icon_color)
    save_icon.config(file= iconPath + r"save.png")
    saveBtn.config(image = save_icon, bg=Icon_color)
    saveAs_icon.config(file= iconPath + r"save as.png")
    saveAsBtn.config(image = saveAs_icon, bg=Icon_color)
    reset_icon.config(file= iconPath + r"reset.png")
    resetBtn.config(image = reset_icon, bg=Icon_color)
    restore_icon.config(file= iconPath + r"restore.png")
    restoreBtn.config(image = restore_icon, bg=Icon_color)
    gray_scale_icon.config(file= iconPath + r"grayscale.png")
    gray_scaleBtn.config(image = gray_scale_icon, bg=Icon_color)
    rotateRIcon.config(file= iconPath + r"rotate clockwise.png")
    rotateRBtn.config(image = rotateRIcon, bg=Icon_color)
    rotateLIcon.config(file= iconPath + r"rotate anticlockwise.png")
    rotateLBtn.config(image = rotateLIcon, bg=Icon_color)
    Move_up.config(file= iconPath + r"arrow up.png")
    transformupBtn.config(image = Move_up, bg=Icon_color)
    Move_left.config(file= iconPath + r"arrow right.png")
    transformleftBtn.config(image = Move_left, bg=Icon_color)
    Move_right.config(file= iconPath + r"arrow left.png")
    transformrightBtn.config(image = Move_right, bg=Icon_color)
    Move_down.config(file= iconPath + r"arrow down.png")
    transformdownBtn.config(image = Move_down, bg=Icon_color)
    Move_up_left.config(file= iconPath + r"arrow down right.png")
    transformulBtn.config(image = Move_up_left, bg=Icon_color)
    Move_up_right.config(file= iconPath + r"arrow down left.png")
    transformurBtn.config(image = Move_up_right, bg=Icon_color)
    Move_down_left.config(file= iconPath + r"arrow up right.png")
    transformdlBtn.config(image = Move_down_left, bg=Icon_color)
    Move_down_right.config(file= iconPath + r"arrow up left.png")
    transformdrBtn.config(image = Move_down_right, bg=Icon_color)
    skewing_R_Transformation.config(file= iconPath + r"skewing right.png")
    skewingRTransformationBtn.config(image = skewing_R_Transformation, bg=Icon_color)
    skewing_L_Transformation.config(file= iconPath + r"skewing left.png")
    skewingLTransformationBtn.config(image = skewing_L_Transformation, bg=Icon_color)
    flip_H.config(file= iconPath + r"flip horizontal.png")
    fliphBtn.config(image = flip_H, bg=Icon_color)
    flip_V.config(file= iconPath + r"flip vertical.png")
    flipvBtn.config(image = flip_V, bg=Icon_color)
    crop_icon.config(file= iconPath + r"crop.png")
    cropBtn.config(image = crop_icon, bg=Icon_color)
    browse_merge_icon.config(file= iconPath + r"Browse.png")
    browse_mergeBtn.config(image = browse_merge_icon, bg=Icon_color)
    merge_icon.config(file= iconPath + r"merge.png")
    mergeBtn.config(image = merge_icon, bg=Icon_color)
    histGraph_icon.config(file= iconPath + r"graph.png")
    histGraphBtn.config(image = histGraph_icon, bg=Icon_color)
    equalHist_icon.config(file= iconPath + r"equalHist.png")
    equalHistBtn.config(image = equalHist_icon, bg=Icon_color)
    negative_icon.config(file= iconPath + r"Negative.png")
    negativeBtn.config(image = negative_icon, bg=Icon_color)
    LogarithmicTrans_icon.config(file= iconPath + r"Logarithmic.png")
    LogarithmicTransBtn.config(image = LogarithmicTrans_icon, bg=Icon_color)
    PowerTrans_icon.config(file= iconPath + r"power.png")
    PowerTransBtn.config(image = PowerTrans_icon, bg=Icon_color)
    Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Bit_PlaneBtn.config(image = Bit_Plane_icon, bg=Icon_color)
    Last_Bit_Plane_icon.config(file= iconPath + r"bit plan.png")
    Last_Bit_PlaneBtn.config(image = Last_Bit_Plane_icon, bg=Icon_color)
    Gray_level_icon.config(file= iconPath + r"gray level.png")
    Gray_levelBtn.config(image = Gray_level_icon, bg=Icon_color)
    sharping1_icon.config(file= iconPath + r"sharpen1.png")
    sharpingBtn1.config(image = sharping1_icon, bg=Icon_color)
    sharping2_icon.config(file= iconPath + r"sharpen2.png")
    sharpingBtn2.config(image = sharping2_icon, bg=Icon_color)
    sharping3_icon.config(file= iconPath + r"sharpen3.png")
    sharpingBtn3.config(image = sharping3_icon, bg=Icon_color)
    edgeBtn_Hor_icon.config(file= iconPath + r"horizontal edge.png")
    edgeBtn_Hor.config(image = edgeBtn_Hor_icon, bg=Icon_color)
    edgeBtn_Ver_icon.config(file= iconPath + r"vertical edge.png")
    edgeBtn_Ver.config(image = edgeBtn_Ver_icon, bg=Icon_color)
    edgeBtn_Diag_icon.config(file= iconPath + r"dignal edge.png")
    edgeBtn_Diag.config(image = edgeBtn_Diag_icon, bg=Icon_color)
    edgeBtn_Sobel_icon.config(file= iconPath + r"sobel.png")
    edgeBtn_Sobel.config(image = edgeBtn_Sobel_icon, bg=Icon_color)
    edgeBtn_laplacian_icon.config(file= iconPath + r"laplacian.png")
    edgeBtn_laplacian.config(image = edgeBtn_laplacian_icon, bg=Icon_color)
    edgeBtn_Canny_icon.config(file= iconPath + r"canny.png")
    edgeBtn_Canny.config(image = edgeBtn_Canny_icon, bg=Icon_color)
    contour_icon.config(file= iconPath + r"contour.png")
    contourBtn.config(image = contour_icon, bg=Icon_color)
    SegmentationKmeans_icon.config(file= iconPath + r"k-means.png")
    SegmentationKmeansBtn.config(image = SegmentationKmeans_icon, bg=Icon_color)
    DetectSimpleGeometricShapes_icon.config(file= iconPath + r"Geometric Shapes.png")
    DetectSimpleGeometricShapesBtn.config(image = DetectSimpleGeometricShapes_icon, bg=Icon_color)
    car_plate_icon.config(file= iconPath + r"car plate.png")
    car_plateBtn.config(image = car_plate_icon, bg=Icon_color)
    Low_pass_filter_icon.config(file= iconPath + r"los pass.png")
    Low_pass_filterBtn.config(image = Low_pass_filter_icon, bg=Icon_color)
    high_pass_filter_icon.config(file= iconPath + r"high pass.png")
    high_pass_filterBtn.config(image = high_pass_filter_icon, bg=Icon_color)
    Lossless_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossless_compression_SaveAsBtn.config(image = Lossless_compression_SaveAs_icon, bg=Icon_color)
    Lossy_compression_SaveAs_icon.config(file= iconPath + r"save as.png")
    Lossy_compression_SaveAsBtn.config(image = Lossy_compression_SaveAs_icon, bg=Icon_color)
    exit_icon.config(file= iconPath + r"exit.png")
    exitBtn.config(image = exit_icon, bg=Icon_color)
    Light_mood_icon.config(file= iconPath + r"light.png")
    Light_moodBtn.config(image = Light_mood_icon, bg=Icon_color)
    Dark_mood_icon.config(file= iconPath + r"dark.png")
    Dark_moodBtn.config(image = Dark_mood_icon, bg=Icon_color)
    Gray_mood_icon.config(file= iconPath + r"gray.png")
    Gray_moodBtn.config(image = Gray_mood_icon, bg=Icon_color)
    Dark_Gray_mood_icon.config(file= iconPath + r"dark gray.png")
    Dark_Gray_moodBtn.config(image = Dark_Gray_mood_icon, bg=Icon_color)

    browse_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    save_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    saveAs_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    reset_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    restore_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    gray_scale_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    rotate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    transform_text.config(fg=Text_Color_font, bg=Text_color_bg)
    skewing_text.config(fg=Text_Color_font,  bg=Text_color_bg)
    flip_text.config(fg=Text_Color_font,bg=Text_color_bg)
    crop_text.config(fg=Text_Color_font, bg=Text_color_bg)
    browse_merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    merge_text.config(fg=Text_Color_font, bg=Text_color_bg)
    histGraph_text.config(fg=Text_Color_font, bg=Text_color_bg)
    equalHist_text.config(fg=Text_Color_font, bg=Text_color_bg)
    negative_text.config(fg=Text_Color_font, bg=Text_color_bg)
    LogarithmicTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    PowerTrans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Last_Bit_Plane_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Gray_level_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping1_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    sharping3_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Hor_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Ver_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Diag_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Sobel_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_laplacian_text.config(fg=Text_Color_font, bg=Text_color_bg)
    edgeBtn_Canny_text.config(fg=Text_Color_font, bg=Text_color_bg)
    contour_text.config(fg=Text_Color_font, bg=Text_color_bg)
    SegmentationKmeans_text.config(fg=Text_Color_font, bg=Text_color_bg)
    DetectSimpleGeometricShapes_text.config(fg=Text_Color_font, bg=Text_color_bg)
    car_plate_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Low_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    high_pass_filter_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossless_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Lossy_compression_SaveAs_text.config(fg=Text_Color_font, bg=Text_color_bg)
    themes_text.config(fg=Text_Color_font,  bg=Text_color_bg )
    exit_text.config(fg=Text_Color_font,  bg=Text_color_bg )


    gamma_text.config(fg=Text_Color_font, bg=Text_color_bg )
    gammaVal.config(fg=Text_Color_font,bg=entity_text_Color)
    Bit_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Bit_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Gray_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Gray_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    blur_text.config(fg=Text_Color_font, bg=Text_color_bg)
    MorphologicalG_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_Canny_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_Canny_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    threshold_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_threshold_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_threshold_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Max_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Max_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    Min_contour_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    Min_contour_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    K_Val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    K_Val.config(fg=Text_Color_font,bg=entity_text_Color)
    d0Scale_text.config(fg=Text_Color_font, bg=Text_color_bg)
    d0Scale.config(fg=Text_Color_font , bg= Text_color_bg)
    # quality_val_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val.config(fg=Text_Color_font,bg=entity_text_Color)
    # quality_val2_text.config(fg=Text_Color_font, bg=Text_color_bg)
    # quality_val2.config(fg=Text_Color_font,bg=entity_text_Color)

    pro.update()

#####################################################################################################

iconPath = r"./Light_Icons/"
Icon_color = 'white'
Text_color_bg = 'white'
Text_Color_font = 'black'
entity_text_Color = 'white'
main_color_bg = 'white'
fr_up_color_bg = 'white'
fr_down_color_bg = 'white'
fr_left_1_color_bg = 'white'
fr_left_2_color_bg = 'white'
fr_left_3_color_bg = "white"
fr2_right_1_color_bg = 'white'
fr2_right_2_color_bg = 'white'
fr2_right_3_color_bg = 'white'

pro = Tk()
pro.title('Zyad Tool Box')
pro.state('zoomed')
# pro.attributes('-fullscreen', True)
pro.config(background= main_color_bg)
pro.iconbitmap(iconPath + r"zyad-logo.ico")

canvas = Canvas(pro, width = 804.5, height = 600 ,bg=main_color_bg)
canvas.place(x=278.4,y=90)

fr_up = Frame(pro , width=807, height=91.5, bg= fr_up_color_bg, border=0)  # Up frame
fr_up.place(x=278.4,y=0)

fr_down = Frame(pro , width=807, height=40, bg= fr_down_color_bg, border=0)  # Down frame
fr_down.place(x=278.4,y=660)

fr_left = Frame(pro , width=280, height=700, border=0)  # Left frame
fr_left.place(x=0, y=0)

fr_left_1 = Frame(fr_left , width=280, height=233.3, bg= fr_left_1_color_bg, border=0)  # Left 1 frame
fr_left_1.place(x=0, y=0)

fr_left_2 = Frame(fr_left , width=280, height=233.3, bg= fr_left_2_color_bg, border=0)  # Left 2 frame
fr_left_2.place(x=0, y=233.3)

fr_left_3 = Frame(fr_left , width=280, height=233.3, bg= fr_left_3_color_bg, border=0)  # Left 3 frame
fr_left_3.place(x=0, y=466.4)

fr2_right = Frame(pro , width=280, height=700, border=0)  # Right frame
fr2_right.place(x=1085, y=0)

fr2_right_1 = Frame(fr2_right , width=280, height=700, bg= fr2_right_1_color_bg, border=0)  # Right 1 frame
fr2_right_1.place(x=0, y=0)

fr2_right_2 = Frame(fr2_right , width=280, height=700, bg= fr2_right_2_color_bg, border=0)  # Right 2 frame
fr2_right_2.place(x=0, y=233.3)

fr2_right_3 = Frame(fr2_right , width=280, height=700, bg= fr2_right_3_color_bg, border=0)  # Right 3 frame
fr2_right_3.place(x=0, y=466.4)

Info= Balloon(pro)

def changeOnBorder(button):
    button.bind("<Enter>", func=lambda e: button.config(border=3))
    button.bind("<Leave>", func=lambda e: button.config(border=1))

#####################################################################################################
arrOfCanvas = 0

def findEmptyCanves(img):
    global arrOfCanvas
    global photo3
    global photo4
    global imageTest_1
    global imageTest_2
    if arrOfCanvas == 0:
        imageTest_1 = img
        if img.shape[0] > 280 or img.shape[1] > 116:
        # if img.shape[0] > 280 or img.shape[1] > 233:
            dim2 = (280,116)
            # dim2 = (280,233)
            img = cv.resize(img,dim2, interpolation = cv.INTER_AREA)
        photo3 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        canvasTest_1.create_image(0, 0, image=photo3, anchor=NW)
    elif arrOfCanvas == 1:
        imageTest_2 = img
        if img.shape[0] > 280 or img.shape[1] > 116:
            dim2 = (280,116)
            img = cv.resize(img,dim2, interpolation = cv.INTER_AREA)
        photo4 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        canvasTest_2.create_image(0, 0, image=photo4, anchor=NW)
    arrOfCanvas = arrOfCanvas + 1
    if arrOfCanvas > 1:
        arrOfCanvas = 0
    return arrOfCanvas

def canvasTest_1Fun(event):
    global photo
    global photo4
    global imageTest_2
    global image
    global arrOfCanvas
    if imageTest_1 != '':
        image = imageTest_1
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0,0,image=photo, anchor=NW)
        # photo4 = ''
        # imageTest_2 = ''
        canvasTest_2.create_image(0,0,image=photo4, anchor=NW)
        arrOfCanvas = 1
canvasTest_1 = Canvas(fr2_right_1, width=280, height=116,bg=fr2_right_1_color_bg)
# canvasTest_1 = Canvas(fr2_right_1, width=280, height=233,bg=fr2_right_1_color_bg)
canvasTest_1.bind("<Button-1>", canvasTest_1Fun)
canvasTest_1.place(x=-2,y=-2)

def canvasTest_2Fun(event):
    global photo
    global image
    if imageTest_2 != '':
        image = imageTest_2
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        canvas.create_image(0,0,image=photo, anchor=NW)
canvasTest_2 = Canvas(fr2_right_1, width=280, height=116,bg=fr2_right_1_color_bg,border=0)
# canvasTest_2 = Canvas(fr2_right_1, width=280, height=116,bg=fr2_right_1_color_bg,border=0)
canvasTest_2.bind("<Button-1>", canvasTest_2Fun)
canvasTest_2.place(x=-2,y=116)

browse_text = Label(fr_up, text= "Browse Image", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=11, height=1 )
browse_text.place(x=15,y=15)
browse_icon = PhotoImage(file= iconPath + r"Browse.png")
browseBtn = Button(fr_up, image = browse_icon, bg=Icon_color, command=Image_Browse )
browseBtn.place(x=50,y=50)
changeOnBorder(browseBtn)

save_text = Label(fr_up, text= "Save Image", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=9, height=1 )
save_text.place(x=155,y=15)
save_icon = PhotoImage(file= iconPath + r"save.png")
saveBtn = Button(fr_up, image = save_icon, bg=Icon_color, command=save)
saveBtn.place(x=185,y=50)
changeOnBorder(saveBtn)

saveAs_text = Label(fr_up, text= "Save As Image", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=12, height=1 )
saveAs_text.place(x=280,y=15)
saveAs_icon = PhotoImage(file= iconPath + r"save as.png")
saveAsBtn = Button(fr_up, image = saveAs_icon, bg=Icon_color, command=saveAs)
saveAsBtn.place(x=320,y=50)
changeOnBorder(saveAsBtn)

reset_text = Label(fr_up, text= "Reset Image", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=10, height=1 )
reset_text.place(x=430,y=15)
reset_icon = PhotoImage(file= iconPath + r"reset.png")
resetBtn = Button(fr_up, image = reset_icon , bg=Icon_color, command=resets)
resetBtn.place(x=460,y=50)
changeOnBorder(resetBtn)
Info.bind_widget(resetBtn, balloonmsg = "Restore the image to original image")

restore_text = Label(fr_up, text= "Restore Image", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=11, height=1 )
restore_text.place(x=565,y=15)
restore_icon = PhotoImage(file= iconPath + r"restore.png")
restoreBtn = Button(fr_up, image = restore_icon, bg=Icon_color, command=restore)
restoreBtn.place(x=603,y=50)
changeOnBorder(restoreBtn)
Info.bind_widget(restoreBtn, balloonmsg = "Restore the image to the last save")

gray_scale_text = Label(fr_up, text= "Gray Scale", fg=Text_Color_font,  bg=Text_color_bg , font = 1 , width=8, height=1 )
gray_scale_text.place(x=700,y=15)
gray_scale_icon = PhotoImage(file= iconPath + r"grayscale.png")
gray_scaleBtn = Button(fr_up, image = gray_scale_icon, bg=Icon_color, command=grayScale)
gray_scaleBtn.place(x=723,y=50)
changeOnBorder(gray_scaleBtn)
Info.bind_widget(gray_scaleBtn, balloonmsg = "change image from BGR to Gray Scale")


rotate_text = Label(fr_left_1, text= "Rotate", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=6, height=1 )
rotate_text.place(x=200,y=5)
rotateRIcon = PhotoImage(file= iconPath + r"rotate clockwise.png")
rotateRBtn = Button(fr_left_1, image=rotateRIcon, bg=Icon_color, command=RotateImgR)
rotateRBtn.place(x=200,y=39)
changeOnBorder(rotateRBtn)
Info.bind_widget(rotateRBtn, balloonmsg = "Rotational image to right by 90")
rotateLIcon = PhotoImage(file= iconPath + r"rotate anticlockwise.png")
rotateLBtn = Button(fr_left_1, image=rotateLIcon, bg=Icon_color, command=RotateImgL)
rotateLBtn.place(x=235,y=39)
changeOnBorder(rotateLBtn)
Info.bind_widget(rotateLBtn, balloonmsg = "Rotational image to left by 90")

transform_text = Label(fr_left_1, text= "Transform", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=8, height=1 )
transform_text.place(x=30,y=5)
Move_up = PhotoImage(file= iconPath + r"arrow up.png")
transformupBtn = Button(fr_left_1, image=Move_up, bg=Icon_color, command=translationTop)
transformupBtn.place(x=55,y=124)
# (x=1290,y=665)
changeOnBorder(transformupBtn)
Info.bind_widget(transformupBtn, balloonmsg = "transform the image to down")
Move_left = PhotoImage(file= iconPath + r"arrow right.png")
transformleftBtn = Button(fr_left_1, image=Move_left, bg=Icon_color, command=translationLeft)
transformleftBtn.place(x=100,y=79)
changeOnBorder(transformleftBtn)
Info.bind_widget(transformleftBtn, balloonmsg = "transform the image to right")
Move_right = PhotoImage(file= iconPath + r"arrow left.png")
transformrightBtn= Button(fr_left_1, image=Move_right, bg=Icon_color, command=translationRight)
transformrightBtn.place(x=10,y=79)
# (x=1245,y=620)
changeOnBorder(transformrightBtn)
Info.bind_widget(transformrightBtn, balloonmsg = "transform the image to left")
Move_down = PhotoImage(file= iconPath + r"arrow down.png")
transformdownBtn = Button(fr_left_1, image=Move_down, bg=Icon_color, command=translationBottom)
transformdownBtn.place(x=55,y=34)
changeOnBorder(transformdownBtn)
Info.bind_widget(transformdownBtn, balloonmsg = "transform the image to up")

Move_up_left = PhotoImage(file= iconPath + r"arrow down right.png")
transformulBtn = Button(fr_left_1, image=Move_up_left, bg=Icon_color, command=translationTopLeft)
transformulBtn.place(x=85,y=109)
changeOnBorder(transformulBtn)
Info.bind_widget(transformulBtn, balloonmsg = "transform the image to down and right")
Move_up_right = PhotoImage(file= iconPath + r"arrow down left.png")
transformurBtn = Button(fr_left_1, image=Move_up_right, bg=Icon_color, command=translationTopRight)
transformurBtn.place(x=25,y=109)
changeOnBorder(transformurBtn)
Info.bind_widget(transformurBtn, balloonmsg = "transform the image to down and lift")
Move_down_left = PhotoImage(file= iconPath + r"arrow up right.png")
transformdlBtn = Button(fr_left_1, image=Move_down_left, bg=Icon_color, command=translationDownLeft)
transformdlBtn.place(x=85,y=49)
changeOnBorder(transformdlBtn)
Info.bind_widget(transformdlBtn, balloonmsg = "transform the image to up and right")
Move_down_right = PhotoImage(file= iconPath + r"arrow up left.png")
transformdrBtn = Button(fr_left_1, image=Move_down_right, bg=Icon_color, command=translationDownRight)
transformdrBtn.place(x=25,y=49)
changeOnBorder(transformdrBtn)
Info.bind_widget(transformdrBtn, balloonmsg = "transform the image to up and lift")

skewing_text = Label(fr_left_1, text= "Skewing", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=7, height=1 )
skewing_text.place(x=197,y=75)
skewing_R_Transformation = PhotoImage(file= iconPath + r"skewing right.png")
skewingRTransformationBtn = Button(fr_left_1, image= skewing_R_Transformation, bg=Icon_color, command=skewingRTransformation)
skewingRTransformationBtn.place(x=200,y=110)
changeOnBorder(skewingRTransformationBtn)
Info.bind_widget(skewingRTransformationBtn, balloonmsg = "skewing to right")
skewing_L_Transformation = PhotoImage(file= iconPath + r"skewing left.png")
skewingLTransformationBtn = Button(fr_left_1, image= skewing_L_Transformation, fg=Text_Color_font, bg=Icon_color, command=skewingLTransformation)
skewingLTransformationBtn.place(x=235,y=110)
changeOnBorder(skewingLTransformationBtn)
Info.bind_widget(skewingLTransformationBtn, balloonmsg = "skewing to left")

flip_text = Label(fr_left_1, text= "Reflecting", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=8, height=1 )
flip_text.place(x=192,y=145)
flip_H = PhotoImage(file= iconPath + r"flip horizontal.png")
fliphBtn = Button(fr_left_1, image=flip_H, bg=Icon_color, command=flip)
fliphBtn.place(x=200,y=180)
changeOnBorder(fliphBtn)
Info.bind_widget(fliphBtn, balloonmsg = "horezontal flip to image")
flip_V = PhotoImage(file= iconPath + r"flip vertical.png")
flipvBtn = Button(fr_left_1, image=flip_V, bg=Icon_color, command=flipVertical)
flipvBtn.place(x=235,y=180)
changeOnBorder(flipvBtn)
Info.bind_widget(flipvBtn, balloonmsg = "vertical flip to image")

crop_text = Label(fr_left_1, text= "Crop", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=4, height=1 )
crop_text.place(x=140,y=5)
crop_icon = PhotoImage(file= iconPath + r"crop.png")
cropBtn = Button(fr_left_1, image = crop_icon, bg=Icon_color, command= myCrop)
cropBtn.place(x=144,y=35)
changeOnBorder(cropBtn)
Info.bind_widget(cropBtn, balloonmsg = "crop the image")

browse_merge_text = Label(fr_left_1, text= "Browse Merge Image", fg=Text_Color_font,  bg=Text_color_bg , width=15, height=1 )
browse_merge_text.place(x=10,y=157)
browse_merge_icon = PhotoImage(file= iconPath + r"Browse.png")
browse_mergeBtn = Button(fr_left_1, image = browse_merge_icon , bg=Icon_color, command=Merge_Image_Browse)
browse_mergeBtn.place(x=50,y=185)
changeOnBorder(browse_mergeBtn)
Info.bind_widget(browse_mergeBtn, balloonmsg = "Browse the image wil blinding with the original image")

merge_text = Label(fr_left_1, text= "Merge", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=5, height=1 )
merge_text.place(x=130,y=155)
merge_icon = PhotoImage(file= iconPath + r"merge.png")
mergeBtn = Button(fr_left_1, image = merge_icon , bg=Icon_color, command=merge)
mergeBtn.place(x=138,y=185)
changeOnBorder(mergeBtn)
Info.bind_widget(mergeBtn, balloonmsg = "blind between original image and browse merge")

histGraph_text = Label(fr_left_2, text= "HistGraph", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=8, height=1 )
histGraph_text.place(x=5,y=5)
histGraph_icon = PhotoImage(file= iconPath + r"graph.png")
histGraphBtn = Button(fr_left_2, image = histGraph_icon, bg=Icon_color, command=histGraph)
histGraphBtn.place(x=25,y=35)
changeOnBorder(histGraphBtn)
Info.bind_widget(histGraphBtn, balloonmsg = "HistGraph is a graph showing intensity value in image")

equalHist_text = Label(fr_left_2, text= "Histogram Equalization", fg=Text_Color_font, font = 1,  bg=Text_color_bg , width=18, height=1 )
equalHist_text.place(x=105,y=5)
equalHist_icon = PhotoImage(file= iconPath + r"equalHist.png")
equalHistBtn = Button(fr_left_2, image = equalHist_icon , bg=Icon_color, command=equalizeHist)
equalHistBtn.place(x=155,y=35)
changeOnBorder(equalHistBtn)
Info.bind_widget(equalHistBtn, balloonmsg = "Histogram Equalization used to enhance contrast")

negative_text = Label(fr_left_2, text= "Negative", fg=Text_Color_font, bg=Text_color_bg , width=7, height=1 )
negative_text.place(x=5,y=70)
negative_icon = PhotoImage(file= iconPath + r"Negative.png")
negativeBtn = Button(fr_left_2, image =  negative_icon, bg=Icon_color, command=negativeTrans)
negativeBtn.place(x=15,y=100)
changeOnBorder(negativeBtn)
Info.bind_widget(negativeBtn, balloonmsg = "Negative Transformation Used for enhancing white or grey detail embedded in dark regions of an image")

LogarithmicTrans_text = Label(fr_left_2, text= "Logarithmic", fg=Text_Color_font, bg=Text_color_bg , width=9, height=1 )
LogarithmicTrans_text.place(x=80,y=70)
LogarithmicTrans_icon = PhotoImage(file= iconPath + r"Logarithmic.png")
LogarithmicTransBtn = Button(fr_left_2, image = LogarithmicTrans_icon , bg=Icon_color, command=LogarithmicTrans)
LogarithmicTransBtn.place(x=100,y=100)
changeOnBorder(LogarithmicTransBtn)
Info.bind_widget(LogarithmicTransBtn, balloonmsg = "Logarithmic Transformation Used to map a narrow range of dark input values into a wider range of output values")

PowerTrans_text = Label(fr_left_2, text= "Power", fg=Text_Color_font, bg=Text_color_bg , width=6, height=1 )
PowerTrans_text.place(x=5,y=135)
PowerTrans_icon = PhotoImage(file= iconPath + r"power.png")
PowerTransBtn = Button(fr_left_2, image = PowerTrans_icon, bg=Icon_color, command=PowerTrans)
PowerTransBtn.place(x=5,y=165)
changeOnBorder(PowerTransBtn)
Info.bind_widget(PowerTransBtn, balloonmsg = "Power Transformation Used to map a narrow range of dark input values into a wider range of output values or vice versa depending on gamma value")

gamma_text = Label(fr_left_2, text= "gamma", fg=Text_Color_font,  bg=Text_color_bg , width=6, height=1 )
gamma_text.place(x=5,y=200)
gammaVal = Entry(fr_left_2, width=4,fg=Text_Color_font,bg=entity_text_Color)
gammaVal.place(x=57,y=202)
Info.bind_widget(gammaVal, balloonmsg = "gamma value between 0 and 0.999 or between 2 and 25")

Bit_Plane_text = Label(fr_left_2, text= "Bit Plane View", fg=Text_Color_font, bg=Text_color_bg , width=11, height=1 )
Bit_Plane_text.place(x=173,y=70)
Bit_Plane_icon = PhotoImage(file= iconPath + r"bit plan.png")
Bit_PlaneBtn = Button(fr_left_2, image = Bit_Plane_icon , bg=Icon_color, command=Bit_Plane)
Bit_PlaneBtn.place(x=195,y=100)
changeOnBorder(Bit_PlaneBtn)
Info.bind_widget(Bit_PlaneBtn, balloonmsg = "show all bit plane in one plot")

Last_Bit_Plane_text = Label(fr_left_2, text= "Select Bit Plane", fg=Text_Color_font, bg=Text_color_bg , width=12, height=1 )
Last_Bit_Plane_text.place(x=163,y=135)
Last_Bit_Plane_icon = PhotoImage(file= iconPath + r"bit plan.png")
Last_Bit_PlaneBtn = Button(fr_left_2, image = Last_Bit_Plane_icon, bg=Icon_color, command=Last_Bit_Plane)
Last_Bit_PlaneBtn.place(x=195,y=165)
changeOnBorder(Last_Bit_PlaneBtn)
Info.bind_widget(Last_Bit_PlaneBtn, balloonmsg = "choose a bit plane to convert the oragnal image")

Bit_Val_text = Label(fr_left_2, text= "Bit Val", fg=Text_Color_font,  bg=Text_color_bg , width=6, height=1 )
Bit_Val_text.place(x=190,y=200)
Bit_Val = Entry(fr_left_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Bit_Val.place(x=243,y=202)
Info.bind_widget(Bit_Val, balloonmsg = "bit value choose of (2,4,8,16,32,64,128)")

Gray_level_text = Label(fr_left_2, text= "Gray level", fg=Text_Color_font, bg=Text_color_bg , width=8, height=1 )
Gray_level_text.place(x=85,y=135)
Gray_level_icon = PhotoImage(file= iconPath + r"gray level.png")
Gray_levelBtn = Button(fr_left_2, image = Gray_level_icon, bg=Icon_color, command=Gray_level)
Gray_levelBtn.place(x=80,y=165)
changeOnBorder(Gray_levelBtn)
Info.bind_widget(Gray_levelBtn, balloonmsg = "Gray level slicing used to highlight gray range of interest to a viewer by one of two ways")

Max_Gray_Val_text = Label(fr_left_2, text= "Max", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Max_Gray_Val_text.place(x=120,y=165)
Max_Gray_Val = Entry(fr_left_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Max_Gray_Val.place(x=152,y=167)
Info.bind_widget(Max_Gray_Val, balloonmsg = "max val used in gray level")

Min_Gray_Val_text = Label(fr_left_2, text= "Min", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Min_Gray_Val_text.place(x=120,y=190)
Min_Gray_Val = Entry(fr_left_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Min_Gray_Val.place(x=152,y=192)
Info.bind_widget(Min_Gray_Val, balloonmsg = "min val used in gray level")

filterSelect = StringVar()
filters = Combobox(fr_left_3,values=('None','Blur 3X3','Gaussian filter', 'Median filter', 'Bilateral filter' ,
                                    'pyramidal','circular','cone')
                            ,state='readonly',textvariable=filterSelect)
filters.place(x=35,y=10)
filters.current(0)
filters.bind("<<ComboboxSelected>>", BlurCombobox)
Info.bind_widget(filters, balloonmsg = "Blur 3X3: smoothing the image by 3x3 \nGaussian filter 3X3 blur: used befor segmentation to get good enhancement \nMedian filter: used to remove noise specifically salt and paper noise \npyramidal blur:image subject to repeated smoothing and subsampling  \ncircular: smoothing the image by 9x9\ncone filter: smoothing the image and increase brightness \nBilateral filter: It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels = Gaussian 9x9 * 9x9 * 3x3")

blur_text = Label(fr_left_3, text= "Blur", fg=Text_Color_font, bg=Text_color_bg)
blur_text.place(x=5,y=10)

sharping1_text = Label(fr_left_3, text= "Low Sharpen", fg=Text_Color_font,  bg=Text_color_bg , width=10, height=1 )
sharping1_text.place(x=5,y=35)
sharping1_icon = PhotoImage(file= iconPath + r"sharpen1.png")
sharpingBtn1 = Button(fr_left_3, image = sharping1_icon, bg=Icon_color, command=sharpingOne)
sharpingBtn1.place(x=25,y=65)
changeOnBorder(sharpingBtn1)
Info.bind_widget(sharpingBtn1, balloonmsg = "Show the sharpness of the image in a simply")

sharping2_text = Label(fr_left_3, text= "Medium Sharpen", fg=Text_Color_font,  bg=Text_color_bg , width=13, height=1 )
sharping2_text.place(x=85,y=35)
sharping2_icon = PhotoImage(file= iconPath + r"sharpen2.png")
sharpingBtn2 = Button(fr_left_3, image = sharping2_icon, bg=Icon_color, command=sharpingTwo)
sharpingBtn2.place(x=110,y=65)
changeOnBorder(sharpingBtn2)
Info.bind_widget(sharpingBtn2, balloonmsg = "Show the sharpness of the image in a Significantly")

sharping3_text = Label(fr_left_3, text= "High Sharpen", fg=Text_Color_font,  bg=Text_color_bg , width=10, height=1 )
sharping3_text.place(x=190,y=35)
sharping3_icon = PhotoImage(file= iconPath + r"sharpen3.png")
sharpingBtn3 = Button(fr_left_3, image = sharping3_icon, bg=Icon_color, command=sharpingThree)
sharpingBtn3.place(x=210,y=65)
changeOnBorder(sharpingBtn3)
Info.bind_widget(sharpingBtn3, balloonmsg = "Show the sharpness of the image in a grossly")


########################################################################
#####################Morphological Transformations######################

# transformSelect = StringVar()
# transformCombobox = Combobox(pro,values=('None','Erosion', 'Dilation', 
#                                         'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat')
#                                         ,state='readonly',textvariable=transformSelect)
# transformCombobox.place(x=1190,y=40)
# transformCombobox.current(0)
# transformCombobox.bind("<<ComboboxSelected>>", TransformCombobox)

# Morphological_text = Label(pro, text= "Morphological Transformations", fg=Text_Color_font, bg=Text_color)
# Morphological_text.place(x=1010,y=40)

transformSelect = StringVar()
transformCombobox = Combobox(fr2_right_2,values=('None','Erosion', 'Dilation', 
                                        'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat')
                                        ,state='readonly',textvariable=transformSelect)
transformCombobox.place(x=92,y=10)
transformCombobox.current(0)
transformCombobox.bind("<<ComboboxSelected>>", TransformComboboxG)
Info.bind_widget(transformCombobox, balloonmsg = "The type of Morphological Transformations: "+"\n"
                                    +"Dilation operation: adds pixels to the boundaries of objects in an image"+"\n"
                                    +"Erosion: uses a structuring element for probing and reducing the shapes contained in the input image."+"\n"
                                    +"Opening: used to restore or recover the original image to the maximum possible extent."+"\n"
                                    +"Closing: used to smoother the contour of the distorted image and fuse back the narrow breaks and long thin gulfs."+"\n"
                                    +"Gradient: is equal to the difference between dilation and erosion of an image."+"\n"
                                    +"and is used in edge detection, segmentation and to find the outline of an object."+"\n"
                                    +"Top Hat: used to enhance bright objects of interest in a dark background"+"\n"
                                    +"Black Hat: used to do the opposite, enhance dark objects of interest in a bright background.")


MorphologicalG_text = Label(fr2_right_2, text= "Morphological", fg=Text_Color_font, bg=Text_color_bg)
MorphologicalG_text.place(x=5,y=10)

########################################################################

edgeBtn_Hor_text = Label(fr_left_3, text= "Horizontal Edge Detect", fg=Text_Color_font,   bg=Text_color_bg , width=17, height=1 )
edgeBtn_Hor_text.place(x=5,y=100)
edgeBtn_Hor_icon = PhotoImage(file= iconPath + r"horizontal edge.png")
edgeBtn_Hor = Button(fr_left_3, image = edgeBtn_Hor_icon, bg=Icon_color, command=edgeDetectHor)
edgeBtn_Hor.place(x=45,y=130)
changeOnBorder(edgeBtn_Hor)
Info.bind_widget(edgeBtn_Hor, balloonmsg = "Horizontal Edge Detect Detects all Horizontal edge pixels")

edgeBtn_Ver_text = Label(fr_left_3, text= "Vertical Edge Detect", fg=Text_Color_font,   bg=Text_color_bg , width=16, height=1 )
edgeBtn_Ver_text.place(x=150,y=100)
edgeBtn_Ver_icon = PhotoImage(file= iconPath + r"vertical edge.png")
edgeBtn_Ver = Button(fr_left_3, image = edgeBtn_Ver_icon, bg=Icon_color, command=edgeDetectVer)
edgeBtn_Ver.place(x=185,y=130)
changeOnBorder(edgeBtn_Ver)
Info.bind_widget(edgeBtn_Ver, balloonmsg = "Vertical Edge Detect Detects all Vertical edge pixels")

edgeBtn_Diag_text = Label(fr_left_3, text= "Diagonal Edge Detect", fg=Text_Color_font,  bg=Text_color_bg , width=15, height=1 )
edgeBtn_Diag_text.place(x=5,y=165)
edgeBtn_Diag_icon = PhotoImage(file= iconPath + r"dignal edge.png")
edgeBtn_Diag = Button(fr_left_3, image = edgeBtn_Diag_icon, bg=Icon_color, command=edgeDetectDiag)
edgeBtn_Diag.place(x=45,y=195)
changeOnBorder(edgeBtn_Diag)
Info.bind_widget(edgeBtn_Diag, balloonmsg = "Diagonal Edge Detect Detects all Diagonal edge pixels")

edgeBtn_Sobel_text = Label(fr_left_3, text= "Sobel Edge Detect", fg=Text_Color_font,  bg=Text_color_bg , width=15, height=1 )
edgeBtn_Sobel_text.place(x=150,y=165)
edgeBtn_Sobel_icon = PhotoImage(file= iconPath + r"sobel.png")
edgeBtn_Sobel = Button(fr_left_3, image = edgeBtn_Sobel_icon, bg=Icon_color, command=edgeDetectSobel)
edgeBtn_Sobel.place(x=185,y=195)
changeOnBorder(edgeBtn_Sobel)
Info.bind_widget(edgeBtn_Sobel, balloonmsg = "Sobel Edge Detect sums up Horizontal,Vertical and Diagonal to get the full edge")

# getMeanAndStandardDeviationBtn = Button(pro, text="Get data", fg=Text_Color_font, bg='white', command=getMeanAndStandardDeviation)
# getMeanAndStandardDeviationBtn.place(x=1100,y=605)
# changeOnBorder(getMeanAndStandardDeviationBtn)
# Info.bind_widget(getMeanAndStandardDeviationBtn, balloonmsg = "show the value of pixels in terminal")

edgeBtn_laplacian_text = Label(fr2_right_2, text= "Edge Detection laplacian", fg=Text_Color_font,  bg=Text_color_bg , width=18, height=1 )
edgeBtn_laplacian_text.place(x=10,y=35)
edgeBtn_laplacian_icon = PhotoImage(file= iconPath + r"laplacian.png")
edgeBtn_laplacian = Button(fr2_right_2, image = edgeBtn_laplacian_icon, bg=Icon_color, command=edgeDetectLaplacian)
edgeBtn_laplacian.place(x=60,y=65)
changeOnBorder(edgeBtn_laplacian)
Info.bind_widget(edgeBtn_laplacian, balloonmsg = "sholud use gusian fillter befor laplac to get great edge")

edgeBtn_Canny_text = Label(fr2_right_2, text= "Edge Detection Canny", fg=Text_Color_font,  bg=Text_color_bg , width=17, height=1 )
edgeBtn_Canny_text.place(x=150,y=35)
edgeBtn_Canny_icon = PhotoImage(file= iconPath + r"canny.png")
edgeBtn_Canny = Button(fr2_right_2, image = edgeBtn_Canny_icon , bg=Icon_color, command=edgeDetectCanny)
edgeBtn_Canny.place(x=160,y=65)
changeOnBorder(edgeBtn_Canny)
Info.bind_widget(edgeBtn_Canny, balloonmsg = "used to get al edge in image")

Max_Canny_Val_text = Label(fr2_right_2, text= "Max", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Max_Canny_Val_text.place(x=195,y=60)
Max_Canny_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Max_Canny_Val.place(x=225,y=60)
Info.bind_widget(Max_Canny_Val, balloonmsg = "Max val used in Canny")

Min_Canny_Val_text = Label(fr2_right_2, text= "Min", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Min_Canny_Val_text.place(x=195,y=85)
Min_Canny_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Min_Canny_Val.place(x=225,y=85)
Info.bind_widget(Min_Canny_Val, balloonmsg = "Min val used in Canny")

thresholdSelect = StringVar()
thresholdCombobox = Combobox(fr2_right_2,values=('None', 'thresh' ,'Binary', 'Inverse Binary',
                                        'Truncated','To-Zero','Inverse To-Zero',
                                        'Otsu')
                                ,state='readonly',textvariable=thresholdSelect)
thresholdCombobox.place(x=65,y=110)
thresholdCombobox.current(0)
thresholdCombobox.bind("<<ComboboxSelected>>", threshold)
Info.bind_widget(thresholdCombobox, balloonmsg ="Threshold: used to convert a grayscale image to a binary image, where the pixels are either 0 or 255. \nBinary: ch\nInverse Binary: \nTruncated: \nTo-Zero: \nInverse To-Zero: \nOtsu")

threshold_text = Label(fr2_right_2, text= "threshold", fg=Text_Color_font, bg=Text_color_bg)
threshold_text.place(x=5,y=110)

Max_threshold_Val_text = Label(fr2_right_2, text= "Max", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Max_threshold_Val_text.place(x=5,y=135)
Max_threshold_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Max_threshold_Val.place(x=35,y=135)
Info.bind_widget(Max_threshold_Val, balloonmsg = "Max val used in threshold")

Min_threshold_Val_text = Label(fr2_right_2, text= "Min", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Min_threshold_Val_text.place(x=75,y=135)
Min_threshold_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Min_threshold_Val.place(x=105,y=135)
Info.bind_widget(Min_threshold_Val, balloonmsg = "Min val used in threshold")

##################################################################################################
#contour

contour_text = Label(fr2_right_2, text= "Contour", fg=Text_Color_font,  bg=Text_color_bg , font= 1 , width=8, height=1 )
contour_text.place(x=150,y=135)
contour_icon = PhotoImage(file= iconPath + r"contour.png")
contourBtn = Button(fr2_right_2, image = contour_icon, bg=Icon_color, command=contour)
contourBtn.place(x=160,y=170)
changeOnBorder(contourBtn)
Info.bind_widget(contourBtn, balloonmsg = "after edge dtection it fills edges with a color to show edges more visible")

# ImageSegmentationContoursBtn = Button(pro, text="Image Segmentation Contours", fg=Text_Color_font, bg=Color7, command=ImageSegmentationContours)
# ImageSegmentationContoursBtn.place(x=1010,y=270)
# changeOnBorder(ImageSegmentationContoursBtn)
# Info.bind_widget(ImageSegmentationContoursBtn, balloonmsg = "")

#anwa3 al countors 

# featuresOfContoursSelect = StringVar()
# featuresOfContoursCombobox = Combobox(pro,values=('None', 'Moments' ,'Area', 'Perimeter',
#                                         'Approximation','Hull','Convexity',
#                                         'Straight Rectangle', 'Rotated Rectangle','Minimum Enclosing Circle',
#                                         'Fitting an Ellipse','Fitting a Line')
#                                 ,state='readonly',textvariable=featuresOfContoursSelect)
# featuresOfContoursCombobox.place(x=1210,y=200)
# featuresOfContoursCombobox.current(0)
# featuresOfContoursCombobox.bind("<<ComboboxSelected>>", featuresOfContours)
#Info.bind_widget(transformCombobox, balloonmsg = "The type of countors")

Max_contour_Val_text = Label(fr2_right_2, text= "Max", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Max_contour_Val_text.place(x=195,y=165)
Max_contour_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Max_contour_Val.place(x=225,y=165)
Info.bind_widget(Max_contour_Val, balloonmsg = "Max val used in contour")

Min_contour_Val_text = Label(fr2_right_2, text= "Min", fg=Text_Color_font,  bg=Text_color_bg , width=3, height=1 )
Min_contour_Val_text.place(x=195,y=190)
Min_contour_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
Min_contour_Val.place(x=225,y=190)
Info.bind_widget(Min_contour_Val, balloonmsg = "Min val used in contour")

################################################################################################

SegmentationKmeans_text = Label(fr2_right_2, text= "Segmentation K-means", fg=Text_Color_font,  bg=Text_color_bg ,  width=18, height=1 )
SegmentationKmeans_text.place(x=5,y=165)
SegmentationKmeans_icon = PhotoImage(file= iconPath + r"k-means.png")
SegmentationKmeansBtn = Button(fr2_right_2, image = SegmentationKmeans_icon, bg=Icon_color, command=SegmentationK_means)
SegmentationKmeansBtn.place(x=45,y=195)
changeOnBorder(SegmentationKmeansBtn)
Info.bind_widget(SegmentationKmeansBtn, balloonmsg = "SegmentationKmeans")

K_Val_text = Label(fr2_right_2, text= "K_Val", fg=Text_Color_font,  bg=Text_color_bg , width=4, height=1 )
K_Val_text.place(x=80,y=205)
K_Val = Entry(fr2_right_2, width=5,fg=Text_Color_font,bg=entity_text_Color)
K_Val.place(x=118,y=205)
Info.bind_widget(K_Val, balloonmsg = "insert K val between 1 and 5")

DetectSimpleGeometricShapes_text = Label(fr2_right_3, text= "Detect Simple Geometric Shapes", fg=Text_Color_font,  bg=Text_color_bg ,  width=24, height=1 )
DetectSimpleGeometricShapes_text.place(x=10,y=10)
DetectSimpleGeometricShapes_icon = PhotoImage(file= iconPath + r"Geometric Shapes.png")
DetectSimpleGeometricShapesBtn = Button(fr2_right_3, image = DetectSimpleGeometricShapes_icon, bg=Icon_color, command=DetectSimpleGeometricShapes)
DetectSimpleGeometricShapesBtn.place(x=85,y=40)
changeOnBorder(DetectSimpleGeometricShapesBtn)
Info.bind_widget(DetectSimpleGeometricShapesBtn, balloonmsg = "Detect simple geometric shapes and detect tle label for each one")


car_plate_text = Label(fr2_right_3, text= "Car Plate", fg=Text_Color_font,  bg=Text_color_bg ,  width=7, height=1 )
car_plate_text.place(x=210,y=10)
car_plate_icon = PhotoImage(file= iconPath + r"car plate.png")
car_plateBtn = Button(fr2_right_3, image = car_plate_icon, bg=Icon_color,command=car_plate)
car_plateBtn.place(x=220,y=40)
changeOnBorder(car_plateBtn)
Info.bind_widget(car_plateBtn, balloonmsg = "to get the number of car plate")

Low_pass_filter_text = Label(fr2_right_3, text= "Low Pass Filter", fg=Text_Color_font,  bg=Text_color_bg ,  width=11, height=1 )
Low_pass_filter_text.place(x=10,y=75)
Low_pass_filter_icon = PhotoImage(file= iconPath + r"los pass.png")
Low_pass_filterBtn = Button(fr2_right_3, image = Low_pass_filter_icon, bg=Icon_color, command=Low_pass_filter)
Low_pass_filterBtn.place(x=35,y=105)
changeOnBorder(Low_pass_filterBtn)
Info.bind_widget(Low_pass_filterBtn, balloonmsg = "Low Pass Filter convert image to Frequency Domain and smoothing the image and convert to spatial")

high_pass_filter_text = Label(fr2_right_3, text= "High Pass Filter", fg=Text_Color_font,  bg=Text_color_bg ,  width=12, height=1 )
high_pass_filter_text.place(x=100,y=75)
high_pass_filter_icon = PhotoImage(file= iconPath + r"high pass.png")
high_pass_filterBtn = Button(fr2_right_3, image = high_pass_filter_icon, bg=Icon_color, command=high_pass_filter)
high_pass_filterBtn.place(x=125,y=105)
changeOnBorder(high_pass_filterBtn)
Info.bind_widget(high_pass_filterBtn, balloonmsg = "High Pass Filter convert image to Frequency Domain and get edge of the image and convert to spatial")

d0Scale_text = Label(fr2_right_3, text= "High ,Low val", fg=Text_Color_font,  bg=Text_color_bg , width=10, height=1 )
d0Scale_text.place(x=195,y=75)
d0Scale = Scale(fr2_right_3, from_=10, to=100, bg= Text_color_bg , fg=Text_Color_font, orient=HORIZONTAL)
d0Scale.place(x=170,y=100)
Info.bind_widget(d0Scale, balloonmsg = "choose the val of high pass filter and Low pass filter")

Lossless_compression_SaveAs_text = Label(fr2_right_3, text= "Lossless compression SaveAs", fg=Text_Color_font,  bg=Text_color_bg ,  width=21, height=1 )
Lossless_compression_SaveAs_text.place(x=10,y=150)
Lossless_compression_SaveAs_icon = PhotoImage(file= iconPath + r"save as.png")
Lossless_compression_SaveAsBtn = Button(fr2_right_3, image = Lossless_compression_SaveAs_icon, bg=Icon_color, command=Lossless_compression_SaveAs)
Lossless_compression_SaveAsBtn.place(x=70,y=180)
changeOnBorder(Lossless_compression_SaveAsBtn)
Info.bind_widget(Lossless_compression_SaveAsBtn, balloonmsg = "compres the image and save as it")

# quality_val_text = Label(fr2_right_3, text= "Quality val", fg=Text_Color_font,  bg=Text_color_bg , width=8, height=1 )
# quality_val_text.place(x=60,y=180)
# quality_val = Entry(fr2_right_3, width=5,fg=Text_Color_font,bg=entity_text_Color)
# quality_val.place(x=75,y=205)
# Info.bind_widget(quality_val, balloonmsg = "")

Lossy_compression_SaveAs_text = Label(fr2_right_3, text= "Lossy compression SaveAs", fg=Text_Color_font,  bg=Text_color_bg ,  width=19, height=1 )
Lossy_compression_SaveAs_text.place(x=140,y=175)
Lossy_compression_SaveAs_icon = PhotoImage(file= iconPath + r"save as.png")
Lossy_compression_SaveAsBtn = Button(fr2_right_3, image = Lossy_compression_SaveAs_icon, bg=Icon_color, command=Lossy_compression_SaveAs)
Lossy_compression_SaveAsBtn.place(x=200,y=200)
changeOnBorder(Lossy_compression_SaveAsBtn)
Info.bind_widget(Lossy_compression_SaveAsBtn, balloonmsg = "compres the image and save as it")

# quality_val2_text = Label(fr2_right_3, text= "Quality val", fg=Text_Color_font,  bg=Text_color_bg , width=8, height=1 )
# quality_val2_text.place(x=180,y=205)
# quality_val2 = Entry(fr2_right_3, width=5,fg=Text_Color_font,bg=entity_text_Color)
# quality_val2.place(x=245,y=205)
# Info.bind_widget(quality_val2, balloonmsg = "")

exit_text = Label(fr_down, text= "Exit", fg=Text_Color_font,  bg=Text_color_bg , font = 1 ,width=3, height=1 )
exit_text.place(x=550,y=5)

exit_icon = PhotoImage(file= iconPath + r"exit.png")
exitBtn = Button(fr_down, image = exit_icon, bg=Icon_color, command=exit)
exitBtn.place(x=600,y=5)
changeOnBorder(exitBtn)
Info.bind_widget(exitBtn, balloonmsg = "exit")

themes_text = Label(fr_down, text= "Themes", fg=Text_Color_font,  bg=Text_color_bg , font = 1 ,width=6, height=1 )
themes_text.place(x=10,y=5)

Light_mood_icon = PhotoImage(file= iconPath + r"light.png")
Light_moodBtn = Button(fr_down, image = Light_mood_icon, bg=Icon_color, command=thm1)
Light_moodBtn.place(x=90,y=5)
changeOnBorder(Light_moodBtn)
Info.bind_widget(Light_moodBtn, balloonmsg = "Light mood")

Dark_mood_icon = PhotoImage(file= iconPath + r"dark.png")
Dark_moodBtn = Button(fr_down, image = Dark_mood_icon, bg=Icon_color, command=thm2)
Dark_moodBtn.place(x=140,y=5)
changeOnBorder(Dark_moodBtn)
Info.bind_widget(Dark_moodBtn, balloonmsg = "Dark mood")

Dark_Gray_mood_icon = PhotoImage(file= iconPath + r"dark gray.png")
Dark_Gray_moodBtn = Button(fr_down, image = Dark_Gray_mood_icon, bg=Icon_color, command=thm3)
Dark_Gray_moodBtn.place(x=190,y=5)
changeOnBorder(Dark_Gray_moodBtn)
Info.bind_widget(Dark_Gray_moodBtn, balloonmsg = "Dark Gray mood")

Gray_mood_icon = PhotoImage(file= iconPath + r"gray.png")
Gray_moodBtn = Button(fr_down, image = Gray_mood_icon, bg=Icon_color, command=thm4)
Gray_moodBtn.place(x=240,y=5)
changeOnBorder(Gray_moodBtn)
Info.bind_widget(Gray_moodBtn, balloonmsg = "Gray mood")

#####################################################################################################

pro.mainloop()

#####################################################################################################