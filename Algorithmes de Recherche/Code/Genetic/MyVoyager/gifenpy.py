from PIL import Image, ImageDraw
import os

class Node:
    def __init__(self, key, degree):
        self.key = key
        self.children = [None] * degree

def new_node(key, degree):
    return Node(key, degree)

def new_trie(degree):
    root = new_node(0, degree)
    nkeys = 0
    for i in range(degree):
        root.children[i] = new_node(i, degree)
    nkeys += 2  # Skip clear code and stop code
    return root, nkeys

def del_trie(root):
    if root is None:
        return
    for child in root.children:
        if child:
            del_trie(child)

class GIF:
    def __init__(self, fname, width, height, palette=None, depth=8, loop=0):
        self.width = width
        self.height = height
        self.frames = []
        self.palette = palette if palette else self.default_palette(depth)
        self.loop = loop
        self.image = Image.new('P', (width, height))
        self.draw = ImageDraw.Draw(self.image)
    
    def default_palette(self, depth):
        # VGA Palette (like in the original C code)
        vga = [
            (0x00, 0x00, 0x00), (0xAA, 0x00, 0x00), (0x00, 0xAA, 0x00), 
            (0xAA, 0x55, 0x00), (0x00, 0x00, 0xAA), (0xAA, 0x00, 0xAA),
            (0x00, 0xAA, 0xAA), (0xAA, 0xAA, 0xAA), (0x55, 0x55, 0x55),
            (0xFF, 0x55, 0x55), (0x55, 0xFF, 0x55), (0xFF, 0xFF, 0x55),
            (0x55, 0x55, 0xFF), (0xFF, 0x55, 0xFF), (0x55, 0xFF, 0xFF),
            (0xFF, 0xFF, 0xFF)
        ]
        # Complete with shades of color for larger depths
        if depth > 4:
            for r in range(6):
                for g in range(6):
                    for b in range(6):
                        vga.append((r*51, g*51, b*51))
            for i in range(1, 25):
                v = i * 255 // 25
                vga.append((v, v, v))
        return vga

    def add_frame(self, delay=100):
        self.frames.append(self.image.copy())
        # Pillow doesn't support setting delay per frame, will use it when saving

    def save(self, fname):
        self.image.save(fname, save_all=True, append_images=self.frames, loop=self.loop, duration=100)

    def draw_line(self, x0, y0, x1, y1, color):
        if x0 < 0 or x0 >= self.width or x1 < 0 or x1 >= self.width:
            raise ValueError("Invalid x values")
        if y0 < 0 or y0 >= self.height or y1 < 0 or y1 >= self.height:
            raise ValueError("Invalid y values")

        # Use Bresenham's line algorithm to draw the line
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.draw.point((x0, y0), fill=color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def clear_frame(self):
        """Clears the current frame."""
        self.image = Image.new('P', (self.width, self.height))  # Recreate a blank image
        self.draw = ImageDraw.Draw(self.image)  # Recreate the drawing context

    def close(self):
        """Saves the GIF file and closes the drawing context."""
        self.save("tour.gif")  # Save the GIF with a default filename, or adjust as needed.
