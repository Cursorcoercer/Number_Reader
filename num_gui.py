import numpy as np
import pickle
import pyglet
import random
import mlpt
import train_nums


class bw_grid():

    def __init__(self, dimen, pos, size):
        self.dim = dimen
        self.pos = pos
        self.size = size
        self.reset()
        self.draw_color = 1
        self.radius = 0.9

    def reset(self):
        self.grid = mlpt.null_array(self.dim[0], self.dim[1], 0.0)

    def to_hex(self, val):
        return 4*3*(int(255*val),)

    def color_list(self):
        colors = ()
        for f in range(self.dim[0]):
            for g in range(self.dim[1]):
                colors += self.to_hex(self.grid[f][g])
        return colors

    def vertex_list(self):
        xsize = self.size[0]/self.dim[0]
        ysize = self.size[1]/self.dim[1]
        verts = ()
        for f in range(self.dim[0]):
            for g in range(self.dim[1]):
                verts += (f*xsize+self.pos[0], g*ysize+self.pos[1],
                      (f+1)*xsize+self.pos[0], g*ysize+self.pos[1],
                      (f+1)*xsize+self.pos[0], (g+1)*ysize+self.pos[1],
                          f*xsize+self.pos[0], (g+1)*ysize+self.pos[1])
        return verts

    def blit(self):
        pyglet.graphics.draw(4*self.dim[0]*self.dim[1], pyglet.gl.GL_QUADS,
                    ('v2f', self.vertex_list()),
                    ('c3B', self.color_list()))

    def draw(self, pos):
        loc = (self.dim[0]*(pos[0]-self.pos[0])/self.size[0],
               self.dim[1]*(pos[1]-self.pos[1])/self.size[1])
        if (not 0 <= loc[0] <= self.dim[0] or
            not 0 <= loc[1] <= self.dim[1]):
            return False
        rad = self.radius**2
        for f in range(self.dim[0]):
            for g in range(self.dim[1]):
                c1 = (f-loc[0])**2 + (g-loc[1])**2 < rad
                c2 = (f+1-loc[0])**2 + (g-loc[1])**2 < rad
                c3 = (f-loc[0])**2 + (g+1-loc[1])**2 < rad
                c4 = (f+1-loc[0])**2 + (g+1-loc[1])**2 < rad
                val = (c1+c2+c3+c4)/4
                self.grid[f][g] = (1-val)*self.grid[f][g] + val*self.draw_color
        return True

    def load(self, data):
        self.grid = np.flip(np.transpose(np.reshape(data, self.dim)), 1)

    def vector_data(self):
        return np.reshape(np.transpose(np.flip(self.grid, 1)), self.dim[0]*self.dim[1])


class button():

    def __init__(self, sprite, size, pos):
        self.sprite = sprite
        self.set_size(size)
        self.set_pos(pos)

    def set_size(self, size):
        self.size = size
        self.sprite.update(scale_x=self.size[0]/self.sprite.image.width,
                           scale_y=self.size[1]/self.sprite.image.height)

    def set_pos(self, pos):
        self.pos = pos
        self.sprite.update(x=pos[0], y=pos[1])

    def blit(self):
        self.sprite.draw()

    def press(self, pos):
        if (self.pos[0] <= pos[0] <= self.pos[0]+self.size[0] and
            self.pos[1] <= pos[1] <= self.pos[1]+self.size[1]):
            return True

class text():

    def __init__(self, st, **kwargs):
        self.kwargs = kwargs
        self.set_text(st)

    def set_text(self, st):
        self.text = pyglet.text.Label(str(st), **self.kwargs)

    def blit(self):
        self.text.draw()


if __name__ == '__main__':
    name = "num_rdr_15"
    file = open("networks/"+name+".pkl", "rb")
    num_rdr = pickle.load(file)
    file.close()
    training_data, validation_data, test_data = train_nums.load_data_wrapper()

    window = pyglet.window.Window(fullscreen=True)
    scsz = window.get_size()
    pyglet.gl.glClearColor(0.3,0.3,0.5,1)
    fps_display = pyglet.window.FPSDisplay(window=window)
    FPS = 60

    grid = bw_grid((28, 28), (scsz[1]//6, scsz[1]//6), (2*scsz[1]//3, 2*scsz[1]//3))

    image = pyglet.image.load("icons/erase.png")
    erase = button(pyglet.sprite.Sprite(image), (scsz[1]//10, scsz[1]//10), (scsz[1], 46*scsz[1]//60))
    image = pyglet.image.load("icons/draw.png")
    draw = button(pyglet.sprite.Sprite(image), (scsz[1]//10, scsz[1]//10), (scsz[1], 38*scsz[1]//60))
    image = pyglet.image.load("icons/trash.png")
    trash = button(pyglet.sprite.Sprite(image), (scsz[1]//10, scsz[1]//10), (scsz[1], 30*scsz[1]//60))
    image = pyglet.image.load("icons/sample.png")
    sample = button(pyglet.sprite.Sprite(image), (scsz[1]//10, scsz[1]//10), (scsz[1], 22*scsz[1]//60))
    buttons = [erase, draw, trash, sample]

    act_num = text('', font_name='Times New Roman',
                       font_size=scsz[1]//10,
                       x=113*scsz[1]//100, y=22*scsz[1]//60)
    num_guess = text('', font_name='Times New Roman',
                         font_size=scsz[1]//5, color=(140, 240, 140, 255),
                         x=(scsz[0]+scsz[1])//2, y=scsz[1]//2,
                         anchor_x='center', anchor_y='center')
    text = [act_num, num_guess]

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            b_states = []
            for b in buttons:
                b_states.append(b.press((x, y)))
            if b_states[0]:
                grid.draw_color = 0
            if b_states[1]:
                grid.draw_color = 1
            if b_states[2]:
                grid.reset()
                act_num.set_text('')
            if b_states[3]:
                ran = random.randrange(len(training_data[0]))
                grid.load(training_data[0][ran])
                act_num.set_text(train_nums.max_ind(training_data[1][ran]))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        pass

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            if grid.draw((x, y)):
                act_num.set_text('')

    @window.event
    def on_draw():
        window.clear()
        grid.blit()
        for b in buttons:
            b.blit()
        for t in text:
            t.blit()
        fps_display.draw()

    def update(dt):
        num_guess.set_text(train_nums.max_ind(num_rdr.evaluate(grid.vector_data())))

    pyglet.clock.schedule_interval(update, 1/FPS)
    pyglet.app.run()





