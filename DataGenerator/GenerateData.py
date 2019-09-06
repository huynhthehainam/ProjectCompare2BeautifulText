from PIL import Image, ImageDraw, ImageFont
import os

def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)
FontName = 'Bahnschrift.ttf'
fnt = ImageFont.truetype(FontName, 15)

text = 'Hai Nam'
FileName  = 'text.png'
width,  height = getSize(text,fnt)
MaxSize = max(width,height)
image = Image.new(mode = "RGB", size = (MaxSize,MaxSize),color=(255,0,0))
draw = ImageDraw.Draw(image)
draw.text((0,0), text, font=fnt, fill=(255,255,0))
image.save(FileName)
print('{},{}'.format(width,height))