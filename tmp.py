"""
This is an example to show how to build cross-GUI applications using
matplotlib event handling to interact with objects on the canvas

"""
import matplotlib
matplotlib.use('Qt4Agg')

import numpy as np
import cv2
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment

class PolygonInteractor(object):
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        poly.set_fill(False)
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker='o', markerfacecolor='r', animated=True)
        self.ax.add_line(self.line)
        #self._update_line(poly)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'

        def print_pts(pts):
            s = "["
            pts2 = []
            for i in range(4):
                s += "({:d}, {:d}), ".format(int(pts[i][0]), int(pts[i][1]))
                pts2.append((int(pts[i][0]), int(pts[i][1])))
            s = s[:-2] + "]"
            print(s)
            return tuple(pts2)

        def CalcDstCoords(botL, topL, topR, botR):
            d_botL = (botL[0], 700)
            d_botR = (botR[0], 700)
            d_topL = (d_botL[0], 50)
            d_topR = (d_botR[0], 50)
            return d_botL, d_topL, d_topR, d_botR

        def transformImage(img, matrix):
            return cv2.warpPerspective(img, matrix, (img.shape[:2])[::-1], flags=cv2.INTER_LINEAR)

        def polylines(im, pts, color=[0, 200, 0], thickness=5, isClosed=True):
            return cv2.polylines(im, pts=np.array([pts], dtype=np.int32), isClosed=isClosed, color=color, thickness=thickness)

        def rect(im, topLeft, sizeWH, color=[0, 200, 0], thickness=5):
            x, y = topLeft
            w, h = sizeWH
            pts = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.int)
            return polylines(im, pts, color=color, thickness=thickness, isClosed=True)

        def project(pts):
            src = np.array(pts)
            dst = np.array(CalcDstCoords(*src))
            matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
            imgNew = transformImage(IMG, matrix)
            rect(imgNew, dst[1], ((dst[2][0] - dst[1][0]),(dst[2][1] - dst[1][1])))
            ax2.imshow(imgNew)
            fig2.canvas.draw()

        newPts = print_pts(self.poly.xy)
        project(newPts)

        if not event.inaxes:
            return


        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.array(
                        list(self.poly.xy[:i]) +
                        [(event.xdata, event.ydata)] +
                        list(self.poly.xy[i:]))
                    self.line.set_data(zip(*self.poly.xy))
                    break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np


IMG = imageio.imread('test_images/straight_lines4.jpg')
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # vid = imageio.get_reader('project_video.mp4')
    # [(205, 671), (606, 439), (672, 438), (1109, 662)]
    # [(284, 654), (616, 441), (690, 442), (1041, 651)]
    # [[284, 654], [619, 437], [685, 437], [1041, 651]]
    xys = np.array([(284, 654), (616, 441), (690, 442), (1041, 651)], dtype=np.int)
    xs = xys[:,0]
    ys = xys[:,1]

    poly = Polygon(list(zip(xs, ys)), animated=True)

    ax.add_patch(poly)
    p = PolygonInteractor(ax, poly)

    ax.imshow(IMG)

    ax2.imshow(IMG)
    plt.show()
    # for i in range(340, 380, 10):
    #     ax.imshow(vid.get_data(i))
    #     plt.show()



    #ax.add_line(p.line)
    ax.set_title('Click and drag a point to move it')
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    plt.show()



