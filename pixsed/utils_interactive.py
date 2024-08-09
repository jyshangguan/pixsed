import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from shapely.geometry import shape
from astropy.visualization import simple_norm
from pixsed.utils import get_mask_polygons, plot_mask_contours
from pixsed.utils import poly_to_xy, polys_to_mask


class MaskBuilder_segment:
    '''
    Build the mask by selecting the segmentation map.

    Usage:
        Select the segments by "shift + left click" in the "Segmentation" panel.

        The newly generated mask will be shown in the Manual mask panel.
    '''

    def __init__(self, data, mask, smap, mask_manual=None, ipy=None, fig=None,
                 axs=None, norm_kwargs=None, verbose=False):
        '''
        Start the builder.
        '''
        self._data = data
        self._mask = mask
        self._smap = smap
        self._verbose = verbose

        if mask_manual is None:
            self._mask_manual = np.zeros_like(mask, dtype=bool)
        else:
            self._mask_manual = mask_manual

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        self._norm = simple_norm(data, **norm_kwargs)

        if ipy is None:
            self._ipy = get_ipython()
            self._ipy.run_line_magic('matplotlib', 'tk')
        else:
            self._ipy = ipy

        if fig is None:
            # Change to tkinter backend
            fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.05, hspace=0.1)

            fig.canvas.mpl_connect('button_press_event', self.on_click)
            fig.canvas.mpl_connect('close_event', self.on_close)
        else:
            assert axs is not None, 'Need to provide axs!'

        ax = axs[0, 0]
        ax.imshow(data, origin='lower', cmap='Greys_r', norm=self._norm)
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()

        # Plot the contours of the masks
        plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        ax.set_title('Data', fontsize=16)

        # Prepare the marker
        self._line, = ax.plot([-10], [-10], color='magenta', lw=0.5)
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)

        ax = axs[0, 1]
        ax.imshow(smap, origin='lower', cmap=smap.cmap, interpolation='nearest')
        ax.set_title('Segmentation', fontsize=16)

        ax = axs[1, 0]
        ax.imshow(mask, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Current mask', fontsize=16)
        ax.text(0.2, -0.15, 'Message:', fontsize=16, fontweight='bold',
                transform=ax.transAxes, ha='right', va='top')
        self._text_m = ax.text(0.25, -0.15, '--', fontsize=16,
                               transform=ax.transAxes, ha='left', va='top')

        ax = axs[1, 1]
        self._show_ma = ax.imshow(self._mask_manual, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Manual mask', fontsize=16)

        self._fig = fig
        self._axs = axs

    def on_click(self, event):
        '''
        Get the segment information on click.
        '''
        if event.key == 'shift':
            ix, iy = event.xdata, event.ydata

            if (ix is not None) & (iy is not None):
                pix_x, pix_y = int(ix), int(iy)
                l = self._smap.data[pix_y, pix_x]
                m = self._mask[pix_y, pix_x] | self._mask_manual[pix_y, pix_x]
                self.print_message(f'x = {ix:.2f}, y = {iy:.2f}, label: {l}, masked: {m}')

                if l == 0:
                    self.print_message('This is the background!', color='red')
                elif m:
                    self.print_message('This segment has been masked!', color='red')
                else:
                    self._mask_new = self._smap.data == l
                    self._mask_manual[self._mask_new] = 1

                    pList = get_mask_polygons(self._mask_new)
                    poly = shape(pList[0])
                    xy_poly = np.c_[poly.exterior.xy]
                    self._line.set_data(xy_poly[:, 0], xy_poly[:, 1])
                    self._line.figure.canvas.draw()

                    self._show_ma.set_data(self._mask_manual)
                    self._show_ma.figure.canvas.draw()
            else:
                self.print_message(f'Bad position: ({ix}, {iy})')

    def on_close(self, event):
        '''
        Action on close.
        '''
        self._ipy.run_line_magic('matplotlib', 'inline')
        self.plot_mask_manual()
        plt.show()

    def plot_mask_manual(self):
        '''
        Plot the added mask.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05)

        ax = axs[0]
        ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=self._norm)
        ax.set_title('Data', fontsize=16)

        ax = axs[1]
        ax.imshow(self._mask_manual, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Manual mask', fontsize=16)

    def print_message(self, message, color='k'):
        '''
        Print the message in the text.

        Parameters
        ----------
        message : string
            The message to show.
        '''
        self._text_m.set_text(f'{message}')
        self._text_m.set_color(color)
        self._text_m.figure.canvas.draw()


class MaskBuilder_draw:
    '''
    Build the mask by drawing the mask.

    Usage:
        Press "d" to enter the "drawing mode"; Press "ESC" to exit.

        In the "drawing mode", use "shift + left click" to draw polygon mask in
        the "Data" panel.

        The newly generated mask will be shown in the "Manual mask" panel.
    '''

    def __init__(self, data, mask, mask_manual=None, ipy=None, fig=None, axs=None,
                 norm_kwargs=None, verbose=False):
        '''
        Start the builder.

        Parameters
        ----------
        data : 2D array
            The image data to be displayed.
        mask : 2D array
            The current mask for reference.

        '''
        self._data = data
        self._mask = mask
        self._verbose = verbose

        if mask_manual is None:
            self._mask_manual = np.zeros_like(mask, dtype=bool)
        else:
            self._mask_manual = mask_manual

        # Mode properties
        self._mode = 'None'  # Interactive mode
        self._click_pos = []  # The list of clicked positions
        self._poly_list = []  # The list of polygons
        self._poly_xy = None  # The xy coordinate of the included polygons

        if norm_kwargs is None:
            norm_kwargs = dict(percent=99.99, stretch='asinh', asinh_a=0.001)
        self._norm = simple_norm(data, **norm_kwargs)

        if ipy is None:
            self._ipy = get_ipython()
            self._ipy.run_line_magic('matplotlib', 'tk')
        else:
            self._ipy = ipy

        if fig is None:
            # Change to tkinter backend
            fig, axs = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
            fig.subplots_adjust(wspace=0.05, hspace=0.1)

            fig.canvas.mpl_connect('button_press_event', self.on_click)
            fig.canvas.mpl_connect('key_press_event', self.on_press)
            fig.canvas.mpl_connect('close_event', self.on_close)
        else:
            assert axs is not None, 'Need to provide axs!'

        ax = axs[0, 0]
        ax.imshow(data, origin='lower', cmap='Greys_r', norm=self._norm)
        xlim = ax.get_xlim();
        ylim = ax.get_ylim()

        # Plot the contours of the masks
        plot_mask_contours(mask, ax=ax, verbose=verbose, color='cyan', lw=0.5)
        ax.set_title('Data', fontsize=16)

        # Prepare the marker
        self._line, = ax.plot([-10], [-10], color='magenta', lw=0.5, marker='+')
        ax.set_xlim(xlim);
        ax.set_ylim(ylim)

        ax = axs[0, 1]
        ax.imshow(mask, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Current mask', fontsize=16)

        ax = axs[1, 0]
        self._show_ma = ax.imshow(self._mask_manual, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Manual mask', fontsize=16)

        ax.text(1.3, 1.0, 'Mode:', fontsize=16, fontweight='bold',
                transform=ax.transAxes, ha='right', va='top')
        ax.text(1.3, 0.8, 'Action:', fontsize=16, fontweight='bold',
                transform=ax.transAxes, ha='right', va='top')
        self._text_m = ax.text(1.35, 1.0, f'{self._mode}', fontsize=16,
                               transform=ax.transAxes, ha='left', va='top')
        self._text_a = ax.text(1.35, 0.8, f'--', fontsize=16,
                               transform=ax.transAxes, ha='left', va='top')

        ax = axs[1, 1]
        ax.axis('off')

        self._fig = fig
        self._axs = axs

    def on_click(self, event):
        '''
        Get the segment information on click.
        '''
        if (event.key == 'shift') & (self._mode == 'Draw'):
            ix, iy = event.xdata, event.ydata
            if (ix is not None) and (iy is not None):
                self._click_pos.append((ix, iy))
                xy = np.array(self._click_pos)

                if self._poly_xy is None:
                    x = xy[:, 0]
                    y = xy[:, 1]
                else:
                    x = np.concatenate([self._poly_xy[0], [np.nan], xy[:, 0]])
                    y = np.concatenate([self._poly_xy[1], [np.nan], xy[:, 1]])

                self._line.set_data(x, y)
                self._line.figure.canvas.draw()
                self.print_action(f'Add point ({ix:.2f}, {iy:.2f})')

            else:
                self.print_action(f'Bad point ({ix}, {iy})', 'red')

    def on_press(self, event):
        '''
        Get the pressed key.
        '''
        if event.key == 'd':
            if self._mode != 'Draw':
                self._mode = 'Draw'
                self.print_mode(color='red')
                self.print_action('Start polygon drawing!')

            else:
                if len(self._click_pos) > 2:
                    poly = dict(type='Polygon', coordinates=[self._click_pos + [self._click_pos[0]]])
                    self._poly_list.append(poly)
                    self._click_pos = []  # reset the click_pos

                    # Update the polygons in the image
                    self.plot_poly_list()
                    self.print_action('Save polygon. You can continue to draw!')

                    # Update the mask
                    self._mask_manual |= polys_to_mask(self._poly_list, self._mask.shape)
                    self._show_ma.set_data(self._mask_manual)
                    self._show_ma.figure.canvas.draw()

                else:
                    self.print_action('Less than 3 recorded positions!\nUse ESC to discard the current drawing.')

        elif event.key == 'escape':
            # FIXME: Need to add action to discard the current drawing

            self._mode = 'None'  # Change back to None
            self._click_pos = []  # reset the click_pos
            self.print_mode()
            self.print_action('ESC')

    def on_close(self, event, filename=None, wcs=None):
        '''
        Action on close.
        '''
        self._ipy.run_line_magic('matplotlib', 'inline')
        self.plot_mask_manual()
        plt.show()

        if filename is not None:
            if wcs is None:
                header = wcs.to_header()
            else:
                header = None
            hdul = fits.HDUList([fits.PrimaryHDU(data=self._mask_manual.astype(int), header=header, do_not_scale_image_data=True)])
            hdul.writeto(filename, overwrite=True)

    def plot_mask_manual(self):
        '''
        Plot the manual mask.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05)

        ax = axs[0]
        ax.imshow(self._data, origin='lower', cmap='Greys_r', norm=self._norm)
        ax.set_title('Data', fontsize=16)

        ax = axs[1]
        ax.imshow(self._mask_manual, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
        ax.set_title('Manual mask', fontsize=16)

    def plot_poly_list(self):
        '''
        Plot the current polygon list.
        '''
        x_all = []
        y_all = []
        for p in self._poly_list:
            xy = poly_to_xy(p)
            x_all.append(xy[:, 0])
            x_all.append([np.nan])
            y_all.append(xy[:, 1])
            y_all.append([np.nan])
        x_all = np.concatenate(x_all[:-1])
        y_all = np.concatenate(y_all[:-1])
        self._poly_xy = [x_all, y_all]
        self._line.set_data(x_all, y_all)
        self._line.figure.canvas.draw()

    def print_mode(self, color='k'):
        '''
        Print the mode in the text.
        '''
        self._text_m.set_text(f'{self._mode}')
        self._text_m.set_color(color)
        self._text_m.figure.canvas.draw()

    def print_action(self, message, color='k'):
        '''
        Print the action in the text.

        Parameters
        ----------
        message : string
            The message to show.
        '''
        self._text_a.set_text(f'{message}')
        self._text_a.set_color(color)
        self._text_a.figure.canvas.draw()
