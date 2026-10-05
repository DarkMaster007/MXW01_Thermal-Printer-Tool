from PIL import Image, ImageDraw, ImageFont


W, H = 384, 200
img = Image.new("1", (W, H), 1)
d = ImageDraw.Draw(img)


font_small = ImageFont.load_default()
font_large = ImageFont.load_default()


# Borders & title
d.rectangle((0, 0, W-1, H-1), outline=0, width=1)
d.text((100, 2), "57x25mm TEST PATTERN (384x200px)", font=font_small, fill=0)


# Checkerboard
for y in range(35, 155, 10):
    for x in range(W-170, W-10, 10):
        if ((x+y)//10) % 2 == 0:
            d.rectangle((x, y, x+9, y+9), fill=0)


# Gradient
for x in range(150):
    for y in range(80):
        if (y % 10) < int((x/150)*10):
            img.putpixel((10+x, 40+y), 0)


d.text((12, 130), "Text samples:", font=font_small, fill=0)
d.text((12, 145), "The quick brown fox jumps over the lazy dog.", font=font_small, fill=0)
d.text((12, 160), "0123456789 !@#$%^&*() []{}", font=font_small, fill=0)
d.text((12, 175), "MXW01 | 384x200 px | 8 px/mm", font=font_small, fill=0)


d.text((90, 24), "HELLO, THERMAL PRINTER!", font=font_large, fill=0)


#img.show()
img.save('test_57x25mm_384x200.png')