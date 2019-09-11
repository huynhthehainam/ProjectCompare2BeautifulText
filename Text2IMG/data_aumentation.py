#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:36:33 2019

@author: ngo
"""
import docx2txt as rdx
from PIL import Image, ImageDraw, ImageFont
import os
import re
import random
from numpy import expand_dims
from matplotlib import pyplot
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#read data from docx and get words
def getTokens(doc_path):
    chars = ['“','”','"','.',', ',':','?','>','<',';',':','[',']','{','}',"''",') ',' (','! ','- ','_ ']
    k_mode = ['normal','upper','lower','title']
    modes = {'normal':str,'upper':str.upper,'lower':str.lower,'title':str.title}
    token_index = {'.':1}
    index_token = {1:'.'}
    for t in os.listdir(doc_path):
        for doc in os.listdir(doc_path+t):
            info = rdx.process(doc_path+t+'/'+doc)
            info_process = re.sub(r'\n+', '\n', info).strip()
            for i in range(len(chars)):
                info_process = info_process.replace(chars[i],' '+chars[i]+' ')
            params = info_process.split('\n')
            #parse text follow mode
            for k_idx in range(len(modes)):
                for param in params:
                    sequences = param.split('.')
                    for sequence in sequences:
                        if len(sequence.strip())==0:
                            pass
                        sequence = modes[k_mode[k_idx]](sequence)
                        for word in sequence.split(' '):
                            word = re.sub(' +', ' ', word)
                            word = word.strip()
                            #'T2 ís china text'
                            if t == 'T1':
                                if len(word) == 0:
                                    pass
                                if word not in token_index and len(word) > 0:
                                    assert len(word) > 0
                                    token_index[word] = len(token_index) + 1
                                    index_token[len(token_index)] = word
                            else:
                                # each char in china as word
                                for char in word:
                                    if len(char) == 0:
                                        pass
                                    if char not in token_index and len(char) > 0:
                                        assert len(char) > 0
                                    token_index[char] = len(token_index) + 1
                                    index_token[len(token_index)] = char
    return [token_index,index_token]

#random sequence
def randSeq(index_token,nb_sequence):
    key_list= list(range(1,len(index_token)+1))
    random.shuffle(key_list)
    #nb_words = list(range(5,25))
    nb_words = [1]   
    sequences = []
    w_seq = []
    for i in range(nb_sequence):
        width = 0
        sequence = ''
        #random.shuffle(key_list)
        random.shuffle(nb_words)
        for idx in range(nb_words[0]):
            #sequence += ' '+index_token[key_list[idx]]
            sequence += ' '+index_token[key_list[i]]
        sequences.append(sequence)
        w_seq.append(width)
    return sequences

def draw_underlined_text(draw, pos, text, font, under, **options):
    #if under = 1, draw underline
    twidth, theight = draw.textsize(text, font=font)
    lx, ly = pos[0], pos[1] + theight
    draw.text(pos, text, font=font, **options)
    if under == 1:
        draw.line((lx+7, ly+1, lx + twidth, ly+1), **options)
        
def text2IMG(sequences):
    bg_path = './Background/'
    save_path = '../Data1/'
    font_path = './Fonts/VN/'
    colors = {
                0:(0, 0, 0),
                1:(255, 0, 0)
             }
    seq_idx = 0
    f_size = 25
    count = 0
    nb_gen = 2
    datagen = ImageDataGenerator(
            zca_whitening=True, featurewise_center=True,featurewise_std_normalization=True,
            rotation_range=0.8, zoom_range=[0.95, 1.05], brightness_range=[0.2,1.0],
            width_shift_range=0.01, height_shift_range=0.1, shear_range=0.01
    )
    for sequence in sequences:
        if not os.path.exists(save_path+str(seq_idx)):
            os.mkdir(save_path+str(seq_idx))
        for bg in os.listdir(bg_path):
            bg_without_ext = os.path.splitext(bg)[0]
            img = Image.open(bg_path+bg)
            for font in os.listdir(font_path):
                f_without_ext = os.path.splitext(font)[0]
                img1 = img.copy()
                draw = ImageDraw.Draw(img1)
                fnt = ImageFont.truetype(font_path+font, f_size,encoding="utf-8")
                width,height = draw.textsize(sequence,font=fnt)
                new_img =img1.resize((width-7,width))
                draw = ImageDraw.Draw(new_img)
                draw_underlined_text(draw, (-7,int((width-height)/2)-2), sequence, font=fnt, fill=colors[random.choice([0,1])],under=random.choice([0,1]))
                new_img = new_img.resize((124,124))
                data = img_to_array(new_img)
                # expand dimension to one sample
                samples = expand_dims(data, 0)
                # prepare iterator
                it = datagen.flow(samples, batch_size=1)
                # generate samples and plot
                for i in range(nb_gen):
                	# define subplot
                	count += 1
                    # generate batch of images
                	batch = it.next()
                	# convert to unsigned integers for viewing
                	image = batch[0].astype('uint8')
                	pyplot.imsave(save_path+str(seq_idx)+'/img-'+str(count)+'-'+bg_without_ext+'-'+f_without_ext+'.png',image)
                print ('generate image-'+ str(count)+ '.png done.')
                #count += 1
        seq_idx += 1
        
    print('done.')
doc_path = './Text/'
nb_sequence = 200
token_index,index_token = getTokens(doc_path)
sequences = randSeq(index_token,nb_sequence)
text2IMG(sequences)
