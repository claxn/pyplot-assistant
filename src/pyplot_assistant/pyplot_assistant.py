# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.cm as cm
from PIL import Image
import random
import re
import numpy as np
import os
from Geotiff import Geotiff
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from collections import OrderedDict
from funs import find_fids
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import datetime


# TODO: refactor conversion with getattr
class PyPlotBase(object):
    def __init__(self, output_dirpath="plots", name="base", format="png", overwrite=False, latex=False,
                 size="10x7.5", title="", xlabel="", ylabel="", xmin=None, ymin=None, xmax=None, ymax=None,
                 xtick_labels=None, xtick_rot=None, colours="random", alpha=1, labels=None, legend_loc=1, fontsize=None,
                 ordered=True):

        """

        :param output_dirpath: str
            absolute directory path where the plot should be saved to.
        :param name: str
            name of the plot (will be also the final filename)
        :param format: str
            the format of the output file (e.g. png, pdf, ...)
        :param overwrite: boolean
            flag if an existing file should be written or not.
        :param latex: boolean
            flag if LaTeX should be used or not (LaTeX environment required!).
        :param size: str
            size of the plot in inches given in the following format '{height}x{width}' (e.g. '7.5x10').
        :param title: str
            title of the plot.
        :param xlabel: str
            x axis label.
        :param ylabel: str
            x axis label.
        :param xmin: float/int
            minimum value of x axis.
        :param ymin: float/int
            minimum value of y axis.
        :param xmax: float/int
            maximum value of x axis.
        :param ymax: float/int
            maximum value of y axis.
        :param xtick_labels: list
            labels of x axis.
        :param xtick_rot: int
            rotation of labels along x axis.
        :param colours: str, list
            drawing colour/s of the plot (multiple values can be given as a list).
        :param alpha: float [0,1]
            opacity of the drawing.
        :param labels: str/list
            label/s of
        :param legend_loc: int [0,9]
            location of legend
        :param fontsize: float/int
            fontsize for text-specific parts of the plot
        """
        # general settings
        self.plt = plt
        self.overwrite = overwrite
        self.data_ordered = ordered
        # plot size settings
        self.size = size
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # plot style settings
        self.colours = colours
        self.alpha = alpha
        self.latex = latex
        self.xtick_rot = xtick_rot
        self.legend_loc = legend_loc
        # labelling
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fontsize = fontsize
        self.labels = labels
        self.xtick_labels = xtick_labels
        # plot and file naming
        self.name = name
        self.format = format
        self.output_dirpath = output_dirpath
        filename = self.name + '.' + self.format
        self.fig_name = os.path.join(self.output_dirpath, filename)

        # check data types
        self.__check_data_types()

        # convert data given as strings
        self.__convert()

    def __repr__(self):
        """
        Class representation as string.
        :return: str
            class dictionary in a 'pretty-print' string format
        """
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def __check_data_types(self):
        if type(self.overwrite) != bool:
            err_message = "Argument 'overwrite' has to be of type 'bool', not '{}'".format(type(self.overwrite))
            raise ValueError(err_message)

        if type(self.data_ordered) != bool:
            err_message = "Argument 'data_ordered' has to be of type 'bool', not '{}'".format(type(self.data_ordered))
            raise ValueError(err_message)

        if type(self.size) not in [str, tuple]:
            err_message = "Argument 'size' has to be of type 'tuple' or 'str', not '{}'".format(type(self.size))
            raise ValueError(err_message)

        if type(self.xmin) not in [int, float, datetime.datetime]:
            err_message = "Argument 'xmin' has to be of type 'int', 'float' or 'datetime.datetime' " \
                          "not '{}'".format(type(self.xmin))
            raise ValueError(err_message)

        if type(self.ymin) not in [int, float, datetime.datetime]:
            err_message = "Argument 'ymin' has to be of type 'int', 'float' or 'datetime.datetime' " \
                          "not '{}'".format(type(self.ymin))
            raise ValueError(err_message)

        if type(self.xmax) not in [int, float, datetime.datetime]:
            err_message = "Argument 'xmax' has to be of type 'int', 'float' or 'datetime.datetime' " \
                          "not '{}'".format(type(self.xmax))
            raise ValueError(err_message)

        if type(self.ymax) not in [int, float, datetime.datetime]:
            err_message = "Argument 'ymax' has to be of type 'int', 'float' or 'datetime.datetime' " \
                          "not '{}'".format(type(self.ymax))
            raise ValueError(err_message)

        if type(self.colours) not in [str, list]:
            err_message = "Argument 'colours' has to be of type 'str' or 'list' not '{}'".format(type(self.colours))
            raise ValueError(err_message)

        if type(self.alpha) not in [int, float]:
            err_message = "Argument 'alpha' has to be of type 'int' or 'float' not '{}'".format(type(self.alpha))
            raise ValueError(err_message)

        if type(self.latex) != bool:
            err_message = "Argument 'latex' has to be of type 'bool', not '{}'".format(type(self.latex))
            raise ValueError(err_message)

        if type(self.xtick_rot) not in [int, float]:
            err_message = "Argument 'xtick_rot' has to be of type 'int' or 'float' not '{}'".format(type(self.xtick_rot))
            raise ValueError(err_message)

        if type(self.legend_loc) != int:
            err_message = "Argument 'legend_loc' has to be of type 'int', not '{}'".format(type(self.legend_loc))
            raise ValueError(err_message)

        if type(self.title) != str:
            err_message = "Argument 'title' has to be of type 'str', not '{}'".format(type(self.title))
            raise ValueError(err_message)

        if type(self.xlabel) != str:
            err_message = "Argument 'xlabel' has to be of type 'str', not '{}'".format(type(self.xlabel))
            raise ValueError(err_message)

        if type(self.ylabel) != str:
            err_message = "Argument 'ylabel' has to be of type 'str', not '{}'".format(type(self.ylabel))
            raise ValueError(err_message)

        if type(self.fontsize) != int:
            err_message = "Argument 'fontsize' has to be of type 'int', not '{}'".format(type(self.fontsize))
            raise ValueError(err_message)

        if type(self.labels) not in [str, list]:
            err_message = "Argument 'labels' has to be of type 'str' or 'list' not '{}'".format(type(self.labels))
            raise ValueError(err_message)

        if type(self.xtick_labels) not in [str, list]:
            err_message = "Argument 'xtick_labels' has to be of type 'str' or 'list' not '{}'".format(type(self.xtick_labels))
            raise ValueError(err_message)

        if type(self.name) != str:
            err_message = "Argument 'name' has to be of type 'str', not '{}'".format(type(self.name))
            raise ValueError(err_message)

        if type(self.format) != str:
            err_message = "Argument 'format' has to be of type 'str', not '{}'".format(type(self.format))
            raise ValueError(err_message)

        if type(self.output_dirpath) != str:
            err_message = "Argument 'output_dirpath' has to be of type 'str', not '{}'".format(type(self.output_dirpath))
            raise ValueError(err_message)


    def __check_str(self, class_attributes):
        for class_attribute in class_attributes:
            class_variable = self.__getattribute__(class_attribute)
            if type(class_variable) != str:
                err_message = "Argument '{}' has to be of type 'str', not '{}'".format(class_attribute,
                                                                                       type(class_variable))
                raise ValueError(err_message)

    def __convert(self):

        if type(self.overwrite) == str:
            self.overwrite = PyPlotBase.str2none(self.overwrite)
            if self.overwrite is not None:
                self.overwrite = self.overwrite.lower() == "true"

        if type(self.data_ordered) == str:
            self.data_ordered = PyPlotBase.str2none(self.data_ordered)
            if self.data_ordered is not None:
                self.data_ordered = self.data_ordered.lower() == "true"

        if type(self.xmin) == str:
            self.xmin = PyPlotBase.str2none(self.xmin)
            if self.xmin is not None:
                self.xmin = self.str2num(self.xmin, nodata_value=None)
                if self.xmin is None:
                    self.xmin = self.str2datetime(self.xmin, nodata_value=None)

        if type(self.xmax) == str:
            self.xmax = PyPlotBase.str2none(self.xmax)
            if self.xmax is not None:
                self.xmax = self.str2num(self.xmax, nodata_value=None)
                if self.xmax is None:
                    self.xmax = self.str2datetime(self.xmax, nodata_value=None)

        if type(self.ymin) == str:
            self.ymin = PyPlotBase.str2none(self.ymin)
            if self.ymin is not None:
                self.ymin = self.str2num(self.ymin, nodata_value=None)
                if self.ymin is None:
                    self.ymin = self.str2datetime(self.ymin, nodata_value=None)

        if type(self.ymax) == str:
            self.ymax = PyPlotBase.str2none(self.ymax)
            if self.ymax is not None:
                self.ymax = self.str2num(self.ymax, nodata_value=None)
                if self.ymax is None:
                    self.ymax = self.str2datetime(self.ymax, nodata_value=None)

        if type(self.alpha) == str:
            self.alpha = PyPlotBase.str2none(self.alpha)
            if self.alpha is not None:
                self.alpha = self.str2num(self.alpha, nodata_value=None)

        if type(self.latex) == str:
            self.latex = PyPlotBase.str2none(self.latex)
            if self.latex is not None:
                self.latex = self.latex.lower() == "true"

        if type(self.xtick_rot) == str:
            self.xtick_rot = PyPlotBase.str2none(self.xtick_rot)
            if self.xtick_rot is not None:
                self.xtick_rot = self.str2num(self.xtick_rot, nodata_value=None)

        if type(self.legend_loc) == str:
            self.legend_loc = PyPlotBase.str2none(self.legend_loc)
            if self.legend_loc is not None:
                self.legend_loc = self.str2num(self.legend_loc, nodata_value=None)

        if type(self.fontsize) == str:
            self.fontsize = PyPlotBase.str2none(self.fontsize)
            if self.fontsize is not None:
                self.fontsize = self.str2num(self.fontsize, nodata_value=None)

        if type(self.size) == str:
            self.size = self.dimstr2tuple(self.size)

        if type(self.colours) == str:
            self.colours = self.str2colours(self.colours)

        if type(self.labels) == str:
            self.labels = self.str2labels(self.labels)

        if type(self.xtick_labels) == str:
            self.labels = self.str2labels(self.xtick_labels)

    def parse(self, data, ordered=True, delimiter=';'):
        self.data_ordered = ordered
        x_data = dict()
        y_data = dict()
        data_counter = 0
        for i, entries in enumerate(data):
            if ordered and (entries[0] not in ['x', 'y']):
                if i%2 == 0:
                    axis = "x"
                else:
                    axis = "y"
                entries = axis + delimiter + entries

            entries = entries.split(delimiter)
            data_id = entries[0]
            data_id_split = data_id.split('_')
            axis = data_id_split[0]
            if len(data_id_split) != 2:
                idx = data_counter
            else:
                idx = data_id_split[1]

            if axis == 'x':
                x_data[idx] = [self.str2num(entry, nodata_value=np.nan) for entry in entries[1:]]
            elif axis == 'y':
                y_data[idx] = [self.str2num(entry, nodata_value=np.nan) for entry in entries[1:]]
                data_counter += 1
            else:
                raise Exception('Wrong tag/axis specification. Use either "x" or "y".')

        # equalise both dictionary
        for i in y_data.keys():
            if i not in x_data.keys():
                x_data[i] = x_data[max(x_data.keys())]

        return x_data, y_data

    @staticmethod
    def rnd_colours(n):
        cmap = plt.cm.get_cmap('hsv', 1000)
        rnd_idxs = random.sample(range(0, 1000), n)

        return [cmap(idx) for idx in rnd_idxs]

    @staticmethod
    def str2num(string, nodata_value=None):
        string = string.strip()
        number = nodata_value
        string = PyPlotBase.str2none(string)
        if string is not None:
            try:
                number = float(string)
            except ValueError:
                number = nodata_value

        return number

    @staticmethod
    def str2datetime(string, nodata_value=None):
        used_time_formats = ["%Y%m%d%H%i%s", "%Y%m%d", "%Y-%m-%d %H:%M:%S"]
        date_time = nodata_value
        for used_time_format in used_time_formats:
            try:
                date_time = datetime.strptime(string, used_time_format)
            except:
                continue

        if not date_time:
            raise Exception('Datetime format unknown.')
        return date_time

    @staticmethod
    def str2none(string):
        if string.strip() in ['None', 'NaN', 'nan', 'inf']:
            return None
        else:
            return string

    @staticmethod
    def str2colours(string, delimiter=';', delimiter_channel=',', left_closure='(', right_closure=')'):
        if delimiter == delimiter_channel:
            raise Exception('Delimiter of the different colours has to be different than the delimiter of the colour channels.')
        str_parts = string.split(delimiter)
        colours = []
        n = len(str_parts)
        for i in range(n):
            if delimiter_channel in str_parts[i]:
                colour = PyPlotBase.colourstr2tuple(str_parts[i], delimiter=delimiter, left_closure=left_closure,
                                                    right_closure=right_closure)
                colours.append(colour)
            else:
                colours.append(str_parts[i])

    @staticmethod
    def colourstr2tuple(colour_str, delimiter=',', left_closure='(', right_closure=')'):
        colour_str = colour_str.replace(left_closure, '').replace(right_closure, '').strip()
        colour_str_parts = colour_str.split(delimiter)
        if len(colour_str_parts) != 3:
            raise Exception('The given colour type is not valid. All three colour channels are needed (RGB).')

        colour = (float(colour_str_parts[0]), float(colour_str_parts[1]), float(colour_str_parts[2]))
        return colour

    @staticmethod
    def str2labels(label_str, delimiter=';'):
        labels = label_str.split(delimiter)
        return labels

    @staticmethod
    def dimstr2tuple(dim_str, delimiter='x'):
        dim_str_parts = dim_str.split(delimiter)
        if len(dim_str_parts) != 2:
            raise Exception('Only two dimensions are allowed.')

        dims = (float(dim_str_parts[0]), float(dim_str_parts[1]))
        return dims

    def show(self):
        self.plt.show()

    def save(self):
        fig_labels = self.plt.get_figlabels()
        for fig_name in fig_labels:
            if not os.path.isdir(os.path.dirname(fig_name)):
                os.makedirs(os.path.dirname(fig_name))
            if os.path.isfile(fig_name) and self.overwrite:
                os.remove(fig_name)
            self.plt.figure(fig_name)
            self.plt.savefig(fig_name, bbox_inches="tight")

    def close(self):
        fig_labels = self.plt.get_figlabels()
        for fig_name in fig_labels:
            self.plt.figure(fig_name)
            self.plt.close()

class PyPlotHist(PyPlotBase):

    def __init__(self, output_dirpath="plots", name="base", format="png", overwrite=False, latex=False,
                 size="10x7.5", title="", xlabel="", ylabel="", xmin=None, ymin=None, xmax=None, ymax=None,
                 xtick_labels=None, xtick_rot=None, colours="random", alpha=1, labels=None, legend_loc=1, fontsize=None,
                 ordered=True,
                 bins=10, normed=False, bin_width=None):
        PyPlotBase.__init__(self, output_dirpath=output_dirpath, name=name, format=format, overwrite=overwrite,
                            latex=latex, size=size, title=title, xlabel=xlabel, ylabel=ylabel, xmin=xmin, ymin=ymin,
                            xmax=xmax, ymax=ymax, xtick_labels=xtick_labels, xtick_rot=xtick_rot, colours=colours,
                            alpha=alpha, labels=labels, legend_loc=legend_loc, fontsize=fontsize, ordered=ordered)
        self.bins = bins
        self.normed = normed
        self.bin_width = bin_width

        # check data types
        self.__check_data_types()

        # convert data given as strings
        self.__convert()


    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def __convert(self):

        if type(self.bins) == str:
            self.bins = int(self.bins)

        if type(self.normed) == str:
            self.normed = self.normed.lower() == "true"

        if type(self.bin_width) == str:
            self.bin_width = float(self.bin_width)

    def __check_data_types(self):

        if type(self.bins) != int:
            err_message = "Argument 'bins' has to be of type 'int', not '{}'".format(type(self.bins))
            raise ValueError(err_message)

        if type(self.normed) != bool:
            err_message = "Argument 'normed' has to be of type 'bool', not '{}'".format(type(self.normed))
            raise ValueError(err_message)

        if type(self.bin_width) not in [int, float]:
            err_message = "Argument 'bin_width' has to be of type 'int' or 'float', not '{}'".format(type(self.bin_width))
            raise ValueError(err_message)

    def plot(self, data):
        dims = self.dimstr2tuple(self.size)
        fig = self.plt.figure(num=self.fig_name, figsize=dims, dpi=80, facecolor='w', edgecolor='k')
        fig.set_size_inches(dims[0], dims[1], forward=True)

        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        if type(data[0]) == str:
            x_data, y_data = self._split_xy(data)
        else:
            y_data = dict()
            for idx in range(len(data)):
                y_data[str(idx)] = data[idx]

        self._split_colors(len(y_data))

        # set axis limits
        if not callable(self.xmin):
            xmin = float(self.xmin)
        else:
            xmin = self.xmin([ii for i in y_data.values() for ii in i])
        if not callable(self.xmax):
            xmax = float(self.xmax)
        else:
            xmax = self.xmax([ii for i in y_data.values() for ii in i])

        if not callable(self.xmin) and not callable(self.xmax):
            self.plt.xlim([xmin, xmax])

        # set bins
        if self.bin_width is not None:
            self.bins = list(np.arange(xmin, xmax + self.bin_width, self.bin_width))

        if self.label != "":
            self._split_labels(len(y_data))
            for idx, plt_idx in enumerate(y_data.keys()):
                self.plt.hist(y_data[plt_idx], normed=self.normed, color=self.color[idx], alpha=float(self.alpha),
                              label=self.label[idx], bins=self.bins)
            self.plt.legend(loc=self.legend_loc)
        else:
            for idx, plt_idx in enumerate(y_data.keys()):
                self.plt.hist(y_data[plt_idx], normed=self.normed, color=self.color[idx], alpha=float(self.alpha),
                              bins=self.bins)



        # set bins



class GeoPlotLine(GeoPlot):
    def __init__(self, title="", xlabel="", ylabel="", xtick_labels=None, xtick_rot=None,
                 xmin=None, ymin=None,
                 xmax=None, ymax=None,
                 type="line", name="noname", format="png", size="10x7.5", latex="False",
                 color="random", alpha=1, label="", file_outpath=".\plots", overwrite=True,
                 linestyle='-', linewidth=1, marker='', markerfacecolor='b', markersize=12, fontsize=None, legend_loc=0,
                 legend_bbox=None):

        GeoPlot.__init__(self, title=title, xlabel=xlabel, ylabel=ylabel, xtick_labels=xtick_labels, xtick_rot=xtick_rot,
                 xmin=xmin, ymin=ymin,xmax=xmax, ymax=ymax,
                 type=type, name=name, format=format, size=size, latex=latex,
                 color=color, alpha=alpha, label=label, file_outpath=file_outpath, overwrite=overwrite, fontsize=fontsize)

        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.markerfacecolor = markerfacecolor
        self.markersize = markersize
        self.legend_loc = legend_loc
        self.legend_bbox = legend_bbox

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self, data):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)

        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        if type(data[0]) == str:
            x_data, y_data = self._split_xy(data)
        elif type(data[0]) == dict or type(data[0]) == OrderedDict:
            x_data = data[0]
            y_data = data[1]
        else:
            x_data = {'1': data[0]}
            y_data = {'1': data[1]}

        self._split_colors(len(y_data))

        plot_handles = []
        if self.label != "":
            self._split_labels(len(y_data))
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    h_plot, = self.plt.plot(x_data[plt_idx], y_data[plt_idx], color=self.color[idx], linestyle=self.linestyle,
                              marker=self.marker, alpha=float(self.alpha), label=self.label[idx], linewidth=self.linewidth)
                else:
                    h_plot, = self.plt.plot(y_data[plt_idx], color=self.color[idx], linestyle=self.linestyle, marker=self.marker,
                              alpha=float(self.alpha), label=self.label[idx], linewidth=self.linewidth)
                plot_handles.append(h_plot)
            #self.plt.legend(loc=self.legend_loc, ncol=4, bbox_to_anchor=self.legend_bbox)
            #self.plt.legend(loc=self.legend_loc, ncol=3, bbox_to_anchor=self.legend_bbox)
        else:
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    self.plt.plot(x_data[plt_idx], y_data[plt_idx], color=self.color[idx], linestyle=self.linestyle,
                              marker=self.marker, alpha=float(self.alpha), linewidth=self.linewidth)
                else:
                    self.plt.plot(y_data[plt_idx], color=self.color[idx], linestyle=self.linestyle, marker=self.marker,
                              alpha=float(self.alpha), linewidth=self.linewidth)

        # set axis limits
        if not None in [self.xmin, self.xmax]:
            if not callable(self.xmin):
                xmin = float(self.xmin)
            else:
                xmin = self.xmin([ii for i in x_data.values() for ii in i])
            if not callable(self.xmax):
                xmax = float(self.xmax)
            else:
                xmax = self.xmax([ii for i in x_data.values() for ii in i])
            if not callable(self.xmin) and not callable(self.xmax):
                self.plt.xlim([xmin, xmax])
        if not None in [self.ymin, self.ymax]:
            if not callable(self.ymin):
                ymin = float(self.ymin)
            else:
                ymin = self.ymin([ii for i in y_data.values() for ii in i])
            if not callable(self.ymax):
                ymax = float(self.ymax)
            else:
                ymax = self.ymax([ii for i in y_data.values() for ii in i])
            if not callable(self.ymin) and not callable(self.ymax):
                self.plt.ylim([ymin, ymax])

        if self.xtick_labels is not None:  # assumes monotonous increasing integer values
           fig.canvas.draw()
           ax = self.plt.gca()
           labels_idx = [int(item.get_text().strip('$')) for item in ax.get_xticklabels() if item.get_text() != ""]
           labels_filtered = []
           for label_idx in labels_idx:
               idx_closest = np.argmin(np.abs(label_idx-np.array(xrange(len(self.xtick_labels)))))
               labels_filtered.append(self.xtick_labels[idx_closest])
           ax.set_xticklabels(labels_filtered)

        if self.xtick_rot is not None:
            self.plt.xticks(rotation=self.xtick_rot)

        #self.plt.gca().xaxis.grid(linewidth=0.7, linestyle='--')
        #self.plt.xlim([datetime.datetime(2015,10,15), x_data['1'][-1]])
        self.plt.xlim([datetime.datetime(2015, 07, 01), datetime.datetime(2017, 10, 01)])
        self.plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        import matplotlib.ticker
        nticks = 10
        self.plt.gca().yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        #self.plt.gca().xaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        self.plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,i: str(int(x))))
        self.plt.grid(linewidth=1.5)
        #labels = ['' for item in plt.gca().get_xticklabels()]
        #plt.gca().set_xticklabels(labels)

        return plot_handles


class GeoPlotStem(GeoPlot):
    def __init__(self, title="", xlabel="", ylabel="", xtick_labels=None, xtick_rot=None,
                 xmin=None, ymin=None,
                 xmax=None, ymax=None,
                 type="line", name="noname", format="png", size="10x7.5", latex="False",
                 color="random", alpha=1, label="", file_outpath=".\plots", overwrite=True,
                 linestyle='-', marker='', markerfacecolor='b', markersize=12, fontsize=None, linewidth=12):

        GeoPlot.__init__(self, title=title, xlabel=xlabel, ylabel=ylabel, xtick_labels=xtick_labels, xtick_rot=xtick_rot,
                 xmin=xmin, ymin=ymin,xmax=xmax, ymax=ymax,
                 type=type, name=name, format=format, size=size, latex=latex,
                 color=color, alpha=alpha, label=label, file_outpath=file_outpath, overwrite=overwrite, fontsize=fontsize)

        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.markerfacecolor = markerfacecolor
        self.markersize = markersize

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self, data):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)

        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        if type(data[0]) == str:
            x_data, y_data = self._split_xy(data)
        else:
            x_data = {'1': data[0]}
            y_data = {'1': data[1]}

        self._split_colors(len(y_data))

        if self.label != "":
            self._split_labels(len(y_data))
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    markerline, stemlines, baseline = self.plt.stem(x_data[plt_idx], y_data[plt_idx],
                                                                    alpha=float(self.alpha), label=self.label[idx])
                    self.plt.setp(markerline, 'markerfacecolor', self.markerfacecolor, 'marker',
                                  self.marker, 'markersize', self.markersize)
                    self.plt.setp(stemlines, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                    self.plt.setp(baseline, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                else:
                    markerline, stemlines, baseline = self.plt.stem(y_data[plt_idx], alpha=float(self.alpha),
                                                                    label=self.label[idx])
                    self.plt.setp(markerline, 'markerfacecolor', self.markerfacecolor, 'marker',
                                  self.marker, 'markersize', self.markersize)
                    self.plt.setp(stemlines, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                    self.plt.setp(baseline, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
            self.plt.legend()
        else:
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    markerline, stemlines, baseline = self.plt.stem(x_data[plt_idx], y_data[plt_idx],
                                                                    alpha=float(self.alpha))
                    self.plt.setp(markerline, 'markerfacecolor', self.markerfacecolor, 'marker',
                                  self.marker, 'markersize', self.markersize)
                    self.plt.setp(stemlines, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                    self.plt.setp(baseline, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                else:
                    markerline, stemlines, baseline = self.plt.stem(y_data[plt_idx], alpha=float(self.alpha))
                    self.plt.setp(markerline, 'markerfacecolor', self.markerfacecolor, 'marker',
                                  self.marker, 'markersize', self.markersize)
                    self.plt.setp(stemlines, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])
                    self.plt.setp(baseline, 'linestyle', self.linestyle, 'linewidth', self.linewidth,
                                  'color', self.color[idx])

        # set axis limits
        if not None in [self.xmin, self.xmax]:
            if not callable(self.xmin):
                xmin = float(self.xmin)
            else:
                xmin = self.xmin([ii for i in x_data.values() for ii in i])
            if not callable(self.xmax):
                xmax = float(self.xmax)
            else:
                xmax = self.xmax([ii for i in x_data.values() for ii in i])
            if not callable(self.xmin) and not callable(self.xmax):
                self.plt.xlim([xmin, xmax])
        if not None in [self.ymin, self.ymax]:
            if not callable(self.ymin):
                ymin = float(self.ymin)
            else:
                ymin = self.ymin([ii for i in y_data.values() for ii in i])
            if not callable(self.ymax):
                ymax = float(self.ymax)
            else:
                ymax = self.ymax([ii for i in y_data.values() for ii in i])
            if not callable(self.ymin) and not callable(self.ymax):
                self.plt.ylim([ymin, ymax])

        if self.xtick_labels != None:  # assumes monotonous increasing integer values
           fig.canvas.draw()
           ax = self.plt.gca()
           labels_idx = [int(item.get_text().strip('$')) for item in ax.get_xticklabels() if item.get_text() != ""]
           labels_filtered = []
           for label_idx in labels_idx:
               idx_closest = np.argmin(np.abs(label_idx-np.array(xrange(len(self.xtick_labels)))))
               labels_filtered.append(self.xtick_labels[idx_closest])
           ax.set_xticklabels(labels_filtered)

        if self.xtick_rot != "None":
            self.plt.xticks(rotation=self.xtick_rot)


class GeoPlotPoint(GeoPlot):
    def __init__(self, title="", xlabel="", ylabel="", xtick_labels=None, xtick_rot=None,
                 xmin=lambda x: np.nanmin(x), ymin=lambda x: np.nanmin(x),
                 xmax=lambda x: np.nanmax(x), ymax=lambda x: np.nanmax(x),
                 type="line", name="noname", format="png", size="10x7.5", latex="False",
                 color="random", alpha=1, label="", file_outpath=".\plots", overwrite=True,
                 markersize=20, marker = 'o', colorbar="False", colorbar_label=None, colorbar_orientation='horizontal',
                 colormap=None, vmin=None, vmax=None, colormap_idxs=None, fontsize=None):
        GeoPlot.__init__(self, title=title, xlabel=xlabel, ylabel=ylabel, xtick_labels=xtick_labels,
                         xtick_rot=xtick_rot, xmin=xmin, ymin=ymin,xmax=xmax, ymax=ymax,
                         type=type, name=name, format=format, size=size, latex=latex,
                         color=color, alpha=alpha, label=label, file_outpath=file_outpath, overwrite=overwrite,
                         fontsize=fontsize)
        self.markersize = markersize
        self.marker = marker
        self.vmin = vmin
        self.vmax = vmax
        self.colormap = colormap
        self.colormap_idxs = colormap_idxs
        self.colorbar = colorbar
        self.colorbar_label = colorbar_label
        self.colorbar_orientation = colorbar_orientation

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self, data):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)


        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        if type(data[0]) == str:
            x_data, y_data = self._split_xy(data)
        elif type(data[0]) == dict:
            x_data = data[0]
            y_data = data[1]
        else:
            x_data = {'1': data[0]}
            y_data = {'1': data[1]}

        if self.colorbar.lower() == "true":
            self.color = self.colormap
            self._split_colors(len(y_data))
            self.colormap = self.color
            self.colormap_idxs = [int(colormap_idx) for colormap_idx in self.colormap_idxs.split(';')]
            cm = LinearSegmentedColormap.from_list(
                'user_defined', self.colormap, N=len(self.colormap))
            self.vmin = 0
            self.vmax = len(self.colormap)
            self.color = [None] * len(y_data.keys())
        else:
            cm = None
            self.colormap_idxs = [None] * len(y_data.keys())
            self._split_colors(len(y_data))


        if self.label != "":
            self._split_labels(len(y_data))
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    scatter_plt = self.plt.scatter(x_data[plt_idx], y_data[plt_idx], s=float(self.markersize), color=self.color[idx],
                              marker=self.marker, alpha=float(self.alpha), label=self.label[idx],
                                                   vmin=self.vmin, vmax=self.vmax, c=self.colormap_idxs[idx], cmap=cm)
                else:
                    scatter_plt = self.plt.scatter(np.arange(len(y_data[plt_idx])), y_data[plt_idx], s=float(self.markersize),
                                     color=self.color[idx], marker=self.marker, alpha=float(self.alpha),
                                     label=self.label[idx], vmin=self.vmin, vmax=self.vmax, c=self.colormap_idxs[idx], cmap=cm)
            self.plt.legend(loc=1) #bbox_to_anchor=(0., 1.02, 1., .11))
        else:
            for idx, plt_idx in enumerate(y_data.keys()):
                if plt_idx in x_data.keys():
                    scatter_plt = self.plt.scatter(x_data[plt_idx], y_data[plt_idx], s=float(self.markersize), color=self.color[idx],
                              marker=self.marker, alpha=float(self.alpha), vmin=self.vmin, vmax=self.vmax, c=self.colormap_idxs[idx], cmap=cm)
                else:
                    scatter_plt = self.plt.scatter(np.arange(len(y_data[plt_idx])), y_data[plt_idx], s=float(self.markersize),
                                     color=self.color[idx], marker=self.marker, alpha=float(self.alpha),
                                                   vmin=self.vmin, vmax=self.vmax, c=self.colormap_idxs[idx], cmap=cm)

        if self.colorbar.lower() == "true":
            colorbar_fig = self.plt.colorbar(scatter_plt, pad=0.05, aspect=50, shrink=0.7,
                                             orientation=self.colorbar_orientation)
            if self.colorbar_label is not None:
                colorbar_fig.set_label(self.colorbar_label)

        if self.xtick_labels is not None:  # assumes monotonous increasing integer values
            fig.canvas.draw()
            ax = self.plt.gca()
            labels_idx = [int(item.get_text()) for item in ax.get_xticklabels() if item.get_text() != ""]
            labels_filtered = []
            for label_idx in labels_idx:
                idx_closest = np.argmin(np.abs(label_idx - np.array(xrange(len(self.xtick_labels)))))
                labels_filtered.append(self.xtick_labels[idx_closest])
            ax.set_xticklabels(labels_filtered)

        if self.fontsize != "None":
            self.plt.xticks(rotation=self.xtick_rot)

        # set axis limits
        if not callable(self.xmin):
            xmin = float(self.xmin)
        else:
            xmin = self.xmin([ii for i in x_data.values() for ii in i])
        if not callable(self.ymin):
            ymin = float(self.ymin)
        else:
            ymin = self.ymin([ii for i in y_data.values() for ii in i])
        if not callable(self.xmax):
            xmax = float(self.xmax)
        else:
            xmax = self.xmax([ii for i in x_data.values() for ii in i])
        if not callable(self.ymax):
            ymax = float(self.ymax)
        else:
            ymax = self.ymax([ii for i in y_data.values() for ii in i])

        if not callable(self.xmin) and not callable(self.xmax):
            self.plt.xlim([xmin, xmax])
        if not callable(self.ymin) and not callable(self.ymax):
            self.plt.ylim([ymin, ymax])

class GeoPlotTiff(GeoPlot):
    def __init__(self):
        GeoPlot.__init__(self)
        self.filename = r'tests\test.tif'
        self.colorbar = None
        self.colorbar_labels = None
        self.colorbar_orientation = 'horizontal'
        self.axis_formatter = None  # 'dm'
        self.im_grid = True

        #red = Color("blue")
        #colors = list(red.range_to(Color("red"), 7))
        #color_palette = colors + colors[1:-1][::-1]
        #self.colorbar_palette = [c.get_rgb() for c in color_palette]
        #print self.colorbar_palette

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)
        axes = self.plt.gca()

        if self.latex.lower == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        im = Geotiff(self.filename)
        x0, dx, dxdy, y0, dydx, dy = im.get_geo_info()
        x1 = x0 + dx * im.cols
        y1 = y0 + dy * im.rows

        extent = [x0, x1, y1, y0]
        extent = list(np.array(extent)/1000.)

        #if self.axis_formatter == 'dm':
        #    axes.xaxis.set_major_formatter(FuncFormatter(self._d2dm))
        #    axes.yaxis.set_major_formatter(FuncFormatter(self._d2dm))

        if self.colorbar or self.color:
            if self.color:
                self._split_colors()
                color_map = mpl.colors.ListedColormap(self.color)
                pil_im = Image.open(self.filename)
                pil_imgray = pil_im.convert('L')
                img = 1. - np.array(list(pil_imgray.getdata(band=0)), float)
                img.shape = (pil_imgray.size[1], pil_imgray.size[0])
                im_fig = self.plt.imshow(img, cmap='Greys', extent=extent, alpha=0.7)  # shrink
            else:
                im_fig = self.plt.imshow(Image.open(self.filename), extent=extent)  # shrink

            colorbar_fig = self.plt.colorbar(im_fig, pad=0.05, aspect=50, shrink=0.7,
                                                 orientation=self.colorbar_orientation)
            if self.colorbar_labels:
                colorbar_labels = self.colorbar_labels.split(';')
                colorbar_fig.ax.get_xaxis().set_ticks([])
                for idx, label in enumerate(colorbar_labels):
                    colorbar_fig.ax.text((idx + 0.5) / float(len(colorbar_labels)), -1, label, ha='center',
                                     va='center', fontsize=18)
                #colorbar_fig.ax.tick_params(labelsize=20)
        else:
            self.plt.imshow(Image.open(self.filename), extent=[x0, x1, y1, y0])

        if (self.xmin is not None) and (self.xmax is not None):
            axes.set_xlim([self.xmin, self.xmax])
        else:
            axes.set_xlim([x0, x1])

        if (self.ymin is not None) and (self.ymax is not None):
            axes.set_ylim([self.ymin, self.ymax])
        else:
            axes.set_ylim([y1, y0])


        #if self.im_grid:
        #    self.plt.grid(linewidth=0.7, linestyle='-')

        if self.overview_loc is not None:
            if None not in [self.xmin, self.ymin]:
                if self.xmin is None:
                    self.xmin = x0
                if self.xmax is None:
                    self.xmax = x1
                if self.ymin is None:
                    self.ymin = y0
                if self.ymax is None:
                    self.ymax = y1

                x_range = self.xmax - self.xmin
                y_range = self.ymax - self.ymin

                # draw image boundaries
                x_l = self.xmin + 0.05 * x_range
                x_r = self.xmin + 0.28 * x_range
                #y_b = self.ymin + (1-0.28) * y_range
                #y_t = self.ymin + (1-0.05) * y_range
                y_b = self.ymin + 0.05 * y_range
                y_t = self.ymin + 0.28 * y_range
                self.plt.plot([x_l, x_l], [y_b, y_t], color='red', linewidth=2.5)
                self.plt.plot([x_l, x_r], [y_t, y_t], color='red', linewidth=2.5)
                self.plt.plot([x_r, x_r], [y_t, y_b], color='red', linewidth=2.5)
                self.plt.plot([x_l, x_r], [y_b, y_b], color='red', linewidth=2.5)

                # draw region of interest boundaries
                x_range_roi = x_r - x_l
                y_range_roi = y_t - y_b
                x_l_roi = x_l + (self.xmin - x0)/float(x1 - x0)*x_range_roi
                x_r_roi = x_l + (self.xmax - x0)/float(x1 - x0)*x_range_roi
                y_b_roi = y_b + (self.ymin - y1)/float(y0 - y1)*y_range_roi
                y_t_roi = y_b + (self.ymax - y1)/float(y0 - y1)*y_range_roi
                self.plt.plot([x_l_roi, x_l_roi], [y_b_roi, y_t_roi], color='red', linewidth=2.5)
                self.plt.plot([x_l_roi, x_r_roi], [y_t_roi, y_t_roi], color='red', linewidth=2.5)
                self.plt.plot([x_r_roi, x_r_roi], [y_t_roi, y_b_roi], color='red', linewidth=2.5)
                self.plt.plot([x_l_roi, x_r_roi], [y_b_roi, y_b_roi], color='red', linewidth=2.5)


    def _d2dm(self, x, i):
        deg = str(str(x).split(".")[0]) + r'$^\circ$ '
        min = str(int((x - float(str(x).split(".")[0])) * 60))
        if len(min) == 1:
            min = '0' + min
        min = min + r'$^\prime$'

        return deg + min

class GeoPlotBar(GeoPlot):
    def __init__(self):
        GeoPlot.__init__(self)
        self.bar_width = 0.8
        self.edgecolor = None

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self, data):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)

        if self.latex.lower == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.title(self.title)

        x_data, y_data = self._split_xy(data)
        self._split_colors(len(y_data))

        x_data_format = []
        if self.label != "":
            self._split_labels(len(y_data))
            for idx, plt_idx in enumerate(y_data.keys()):
                self.plt.bar(x_data[plt_idx], y_data[plt_idx], color=self.color[idx], edgecolor=self.edgecolor,
                             alpha=float(self.alpha), label=self.label[idx], align='center')
                x_data_format += x_data[plt_idx]
            self.plt.legend()
        else:
            for idx, plt_idx in enumerate(y_data.keys()):
                self.plt.bar(x_data[plt_idx], y_data[plt_idx], color=self.color[idx], edgecolor=self.edgecolor,
                             alpha=float(self.alpha), align='center')
                x_data_format += x_data[plt_idx]

        if self.xtick_labels is not None:
            axes = self.plt.gca()
            xtick_labels = self.xtick_labels.split(',')

            # Formatter function
            def format_xticks(tick_val, tick_pos):
                if int(tick_val) in list(set(x_data_format)):
                    return xtick_labels[int(tick_val)]
                else:
                    return ''

            axes.xaxis.set_major_formatter(FuncFormatter(format_xticks))

        if self.xtick_rot is not None:
            self.plt.xticks(rotation=int(self.xtick_rot))

        # set axis limits
        if not callable(self.xmin):
            xmin = float(self.xmin)
        else:
            xmin = self.xmin([ii for i in x_data.values() for ii in i])
        if not callable(self.ymin):
            ymin = float(self.ymin)
        else:
            ymin = self.ymin([ii for i in y_data.values() for ii in i])
        if not callable(self.xmax):
            xmax = float(self.xmax)
        else:
            xmax = self.xmax([ii for i in x_data.values() for ii in i])
        if not callable(self.ymax):
            ymax = float(self.ymax)
        else:
            ymax = self.ymax([ii for i in y_data.values() for ii in i])
        self.plt.xlim([xmin, xmax])
        self.plt.ylim([ymin, ymax])


class GeoPlotArray(GeoPlot):
    def __init__(self, title="", xlabel="", ylabel="", xtick_labels=None, xtick_rot=None,
                 xmin=lambda x: np.nanmin(x), ymin=lambda x: np.nanmin(x),
                 xmax=lambda x: np.nanmax(x), ymax=lambda x: np.nanmax(x),
                 type="line", name="noname", format="png", size="10x7.5", latex="False",
                 color="random", alpha=1, label="", file_outpath=".\plots", overwrite=True,
                 array=np.zeros((10, 10)), marker = 'o', colorbar=None, colorbar_label=None, colorbar_orientation='horizontal',
                 colormap=None, vmin=None, vmax=None, colormap_idxs=None, fontsize=None, extent=None):
        GeoPlot.__init__(self, title=title, xlabel=xlabel, ylabel=ylabel, xtick_labels=xtick_labels,
                         xtick_rot=xtick_rot, xmin=xmin, ymin=ymin,xmax=xmax, ymax=ymax,
                         type=type, name=name, format=format, size=size, latex=latex,
                         color=color, alpha=alpha, label=label, file_outpath=file_outpath, overwrite=overwrite,
                         fontsize=fontsize)
        self.array = array
        self.extent = extent
        self.colormap = colormap
        self.colorbar = colorbar
        self.colorbar_labels = colorbar_label
        self.colorbar_orientation = colorbar_orientation

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)
        axes = self.plt.gca()

        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        #self.plt.title(self.title)

        if self.colorbar or self.colormap:
            if self.colormap:
                #self._split_colors()
                im_fig = self.plt.imshow(self.array, cmap=plt.get_cmap(self.colormap), extent=self.extent)  # shrink
            else:
                im_fig = self.plt.imshow(self.array, extent=self.extent)  # shrink

            #self.plt.grid(linewidth=0.7, linestyle='-')
            cbaxes = fig.add_axes([0.215, 0.94, 0.6, 0.03])
            cbaxes.set_title(self.title)
            colorbar_fig = self.plt.colorbar(im_fig, orientation=self.colorbar_orientation, cax=cbaxes)
            #self.plt.gca().grid(False)

            if self.colorbar_labels:
                colorbar_labels = self.colorbar_labels.split(';')
                colorbar_fig.ax.get_xaxis().set_ticks([])
                for idx, label in enumerate(colorbar_labels):
                    colorbar_fig.ax.text((idx + 0.5) / float(len(colorbar_labels)), -0.5, label, ha='center',
                                     va='center')
        elif self.color:
            self._split_colors()
            cm = LinearSegmentedColormap.from_list(
                'user_defined', self.color, N=len(self.color))
            img = plt.imshow(self.array, interpolation='nearest', cmap=cm, extent=self.extent)
            patches = [mpatches.Patch(color=self.color[i], label=self.label[i]) for i in range(len(self.color))]
            plt.legend(handles=patches, bbox_to_anchor=(0., 0.98, 1., .11), loc=9, borderaxespad=0., ncol=3)
            #self.plt.grid(linewidth=0.7, linestyle='-')
            #im = plt.imshow(self.array, interpolation='none')
            #colors = [im.cmap(im.norm(value)) for value in values]
            # create a patch (proxy artist) for every color
            #patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in
            #           range(len(values))]
            # put those patched as legend-handles into the legend
            #plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            self.plt.imshow(self.array)

        if self.xmin and self.xmax:
            axes.set_xlim([self.xmin, self.xmax])

        if self.ymin and self.ymax:
            axes.set_ylim([self.ymin, self.ymax])

class GeoPlotMatrix(GeoPlot):
    def __init__(self, title="", xlabel="", ylabel="", xtick_labels=None, xtick_rot=None,
                 xmin=lambda x: np.nanmin(x), ymin=lambda x: np.nanmin(x),
                 xmax=lambda x: np.nanmax(x), ymax=lambda x: np.nanmax(x),
                 type="line", name="noname", format="png", size="10x7.5", latex="False",
                 color="random", alpha=1, label="", file_outpath=".\plots", overwrite=True,
                 array=np.zeros((10, 10)), marker = 'o', colorbar=None, colorbar_label=None, colorbar_orientation='horizontal',
                 colormap=None, vmin=None, vmax=None, colormap_idxs=None, fontsize=None, ytick_labels=None, with_text=False):
        GeoPlot.__init__(self, title=title, xlabel=xlabel, ylabel=ylabel, xtick_labels=xtick_labels,
                         xtick_rot=xtick_rot, xmin=xmin, ymin=ymin,xmax=xmax, ymax=ymax,
                         type=type, name=name, format=format, size=size, latex=latex,
                         color=color, alpha=alpha, label=label, file_outpath=file_outpath, overwrite=overwrite,
                         fontsize=fontsize,)
        self.array = array
        self.colormap = colormap
        self.colorbar = colorbar
        self.colorbar_labels = colorbar_label
        self.colorbar_orientation = colorbar_orientation
        self.ytick_labels = ytick_labels
        self.vmin = vmin
        self.vmax = vmax
        self.with_text = with_text

    def __repr__(self):
        output = []
        for k in self.__dict__:
            output.append('{0}: {1}'.format(k, self.__dict__[k]))
        return '\n'.join(output)

    def plot(self):
        dims = [float(x) for x in self.size.split('x')]
        fig = self.plt.figure(self.fig_name)
        fig.set_size_inches(dims[0], dims[1], forward=True)
        axes = self.plt.gca()

        if self.latex.lower() == "true":
            self.plt.rc('text', usetex=True)
            self.plt.rc('font', family='serif')
        else:
            self.plt.rc('text', usetex=False)

        if self.fontsize is not None:
            self.plt.rcParams.update({'font.size': self.fontsize})

        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        #self.plt.title(self.title)
        self.plt.xticks(np.linspace(0.5, self.array.shape[1] + 0.5, self.array.shape[1]+1), self.xtick_labels)
        self.plt.yticks(np.linspace(0.5, self.array.shape[1] + 0.5, self.array.shape[1] + 1), self.ytick_labels)
        #self.plt.xticks(np.arange(1, self.array.shape[1], dtype=np.int))
        #self.plt.yticks(np.arange(1, self.array.shape[0], dtype=np.int))
        extent = [0,self.array.shape[1], 0, self.array.shape[0]]

        if self.colorbar or self.colormap:
            if self.colormap:
                #self._split_colors()
                im_fig = self.plt.imshow(self.array, cmap=plt.get_cmap(self.colormap), extent=extent,
                                         interpolation='none',vmin=self.vmin, vmax=self.vmax)  # shrink
            else:
                im_fig = self.plt.imshow(self.array, extent=extent,
                                         interpolation='none',vmin=self.vmin, vmax=self.vmax)  # shrink

            #colorbar_fig = self.plt.colorbar(im_fig, orientation=self.colorbar_orientation)


            if self.colorbar_labels:
                colorbar_labels = self.colorbar_labels.split(';')
                colorbar_fig.ax.get_xaxis().set_ticks([])
                for idx, label in enumerate(colorbar_labels):
                    colorbar_fig.ax.text((idx + 0.5) / float(len(colorbar_labels)), -0.5, label, ha='center',
                                     va='center')
        elif self.color:
            self._split_colors()
            cm = LinearSegmentedColormap.from_list(
                'user_defined', self.color, N=len(self.color))
            img = plt.imshow(self.array, interpolation='nearest', cmap=cm, extent=self.extent)
            patches = [mpatches.Patch(color=self.color[i], label=self.label[i]) for i in range(len(self.color))]
            plt.legend(handles=patches, bbox_to_anchor=(0., 0.98, 1., .11), loc=9, borderaxespad=0., ncol=3)
            self.plt.grid(linewidth=0.7, linestyle='-')
            #im = plt.imshow(self.array, interpolation='none')
            #colors = [im.cmap(im.norm(value)) for value in values]
            # create a patch (proxy artist) for every color
            #patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in
            #           range(len(values))]
            # put those patched as legend-handles into the legend
            #plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            self.plt.imshow(self.array)

        x_minor_locs = np.arange(1, self.array.shape[1], dtype=np.int)
        y_minor_locs = np.arange(1, self.array.shape[0], dtype=np.int)
        #y_minor_locs = np.linspace(0.5, self.array.shape[0], self.array.shape[0] - 1)
        ax = self.plt.gca()
        ax.set_xticks(x_minor_locs, minor=True)
        ax.set_yticks(y_minor_locs, minor=True)
        self.plt.grid(linewidth=2, linestyle='-', which='minor')
        ax.axhline(y=self.array.shape[0], linewidth=4, color='black')
        ax.axhline(y=0, linewidth=4, color='black')
        ax.axvline(x=self.array.shape[1], linewidth=4, color='black')
        ax.axvline(x=0, linewidth=4, color='black')
        ax.tick_params(length=0)

        #if self.xtick_labels is not None:  # assumes monotonous increasing integer values
        #    fig.canvas.draw()
        #    ax = self.plt.gca()
        #    labels_idx = [int(item.get_text()) for item in ax.get_xticklabels() if item.get_text() != ""]
        #    labels_filtered = []
        #    for label_idx in labels_idx:
        #        idx_closest = np.argmin(np.abs(label_idx - np.array(xrange(len(self.xtick_labels)))))
        #        labels_filtered.append(self.xtick_labels[idx_closest])
        #    ax.set_xticklabels(labels_filtered)



        if self.xmin and self.xmax:
            axes.set_xlim([self.xmin, self.xmax])

        if self.ymin and self.ymax:
            axes.set_ylim([self.ymin, self.ymax])

        if self.xtick_rot is not None:
            self.plt.xticks(rotation=int(self.xtick_rot), ha='right')

        # finally, place text if requested
        if self.with_text:
            for row in range(self.array.shape[0]):
                for col in range(self.array.shape[1]):
                    plt.annotate("{:2.2f}".format(self.array[row, col]), xy=(col+0.5, self.array.shape[0] - (row+0.5)),
                                 ha='center', va='center')
                    #plt.annotate("{:2.2f}".format(self.array[row, col]),
                    #             xy=(col + 0.5, self.array.shape[0] - (row + 0.5)),
                    #             ha='center', va='center', fontsize=16, weight='bold')

class GeoPlotReader(GeoPlotHist, GeoPlotLine, GeoPlotPoint, GeoPlotTiff, GeoPlotBar):
    def __init__(self, header):
        attributes = self.read_attr_from_header(header)
        if 'type' in attributes.keys():
            if attributes['type'] == 'hist':
                GeoPlotHist.__init__(self)
            elif attributes['type'] == 'line':
                GeoPlotLine.__init__(self)
            elif attributes['type'] == 'point':
                GeoPlotPoint.__init__(self)
            elif attributes['type'] == 'tiff':
                GeoPlotTiff.__init__(self)
            elif attributes['type'] == 'bar':
                GeoPlotBar.__init__(self)
            else:
                raise Exception('Plot type {0} is not supported.'.format(self.type))
        else:
            GeoPlotHist.__init__(self)

        self.set_attr_from_dict(attributes)
        self.fig_name = os.path.join(self.file_outpath, self.name + '.' + self.format) # create label for figure IDs

    def plot(self, data=None):
        if self.type == 'hist':
            GeoPlotHist.plot(self, data)
        elif self.type == 'line':
            GeoPlotLine.plot(self, data)
        elif self.type == 'point':
            GeoPlotPoint.plot(self, data)
        elif self.type == 'tiff':
            GeoPlotTiff.plot(self)
        elif self.type == 'bar':
            GeoPlotBar.plot(self, data)
        else:
            GeoPlotHist.plot(self, data)

    @staticmethod
    def read_attr_from_header(header):
        attributes = dict()
        for header_line in header:
            attr_val = header_line.split(':') #TODO take only first as attribute
            attr = attr_val[0].strip()
            val = ':'.join(attr_val[1:]).strip()
            if val == "None":
                val = None
            attributes[attr] = val
        return attributes

    @staticmethod
    def read_data(plt_filename):
        with open(plt_filename, 'r') as file_handle:
            content = file_handle.readlines()
        sep_counter = 0
        data_dict = OrderedDict()
        data_dict_counter = 0
        header_curr = []
        data_curr = []
        head_sep = ''.join(['#'] * 100)
        for idx, line in enumerate(content):
            if head_sep == line.strip():
                sep_counter += 1
                if ((sep_counter % 2) != 0) and (sep_counter != 1):
                    tmp_dict = {}
                    tmp_dict['header'] = header_curr
                    tmp_dict['data'] = data_curr
                    data_dict[data_dict_counter] = tmp_dict
                    data_dict_counter += 1
                    header_curr = []
                    data_curr = []
            else:
                if (sep_counter % 2) == 0:
                    data_curr.append(line.strip())
                else:
                    header_curr.append(line.strip())

        tmp_dict = {}
        tmp_dict['header'] = header_curr
        tmp_dict['data'] = data_curr
        data_dict[data_dict_counter] = tmp_dict

        return data_dict

    def set_attr_from_dict(self, attributes):
        for attr in attributes.keys():
            setattr(self, attr, attributes[attr])

    @classmethod
    def from_plt_file(cls, plt_filename, save=True, show=False):
        data_dict = GeoPlotReader.read_data(plt_filename)
        gpr = None
        for data_key in data_dict.keys():
            gpr = cls(data_dict[data_key]['header'])
            #if len(data_dict[data_key]['data']) == 0:
            #    continue
            gpr.plot(data=data_dict[data_key]['data'])
            #if gpr.type == 'point':
            #    gpr.close()

        if gpr is not None:
            if show:
                gpr.show()
            if save:
                gpr.save()
            gpr.close()

class GeoPlotWriter(GeoPlotHist, GeoPlotLine, GeoPlotPoint, GeoPlotTiff, GeoPlotBar):
    def __init__(self, input_filepath, type):
        if type == 'hist':
            GeoPlotHist.__init__(self)
        elif type == 'line':
            GeoPlotLine.__init__(self)
        elif type == 'point':
            GeoPlotPoint.__init__(self)
        elif type == 'tiff':
            GeoPlotTiff.__init__(self)
        elif type == 'bar':
            GeoPlotBar.__init__(self)
        else:
            raise Exception('Plot type {0} is not supported.'.format(type))
        self.type = type
        self.input_filepath = input_filepath

    def write_plt_file(self, data):
        head_sep = ''.join(['#'] * 100)
        keys_ignore = ['input_filepath', 'plt', 'fig_name']

        if os.path.isfile(self.input_filepath) and self.overwrite:
            self.del_plt_file()

        with open(self.input_filepath, 'a+') as input_file:
            # header
            input_file.write(head_sep + '\n')
            for key in self.__dict__.keys():
                if key in keys_ignore or callable(self.__dict__[key]):
                    continue
                input_file.write('{0}: {1} \n'.format(key, self.__dict__[key]))
            input_file.write(head_sep + '\n')
            # data
            for data_row in data:
                input_file.write(data_row + '\n')

    def del_plt_file(self):
        os.remove(self.input_filepath)


def save_plts(plt_filepath):
    plt_fids = find_fids(plt_filepath)
    for plt_fid in plt_fids:
        print('Saved {} ...'.format(os.path.basename(plt_fid)))
        GeoPlotReader.from_plt_file(plt_fid, save=True)

if __name__ == "__main__":
    plt_filename = r'D:\TU_Wien\Master\Masterthesis\Data\plt\environ_bs.plt'
    #GeoPlotReader.from_plt_file(plt_filename)
    #with open('tests/test.plt', 'r') as file_handle:
    #with open('../../data/plt/latlon.plt', 'r') as file_handle:
    with open(r'D:\TU_Wien\Master\Masterthesis\Data\plt\bs_plia\VV_4877550_1590366.plt', 'r') as file_handle:
        content = file_handle.readlines()
    sep_counter = 0
    header_curr = []
    data_curr = []
    head_sep = ''.join(['#'] * 100)
    for idx, line in enumerate(content):
        if head_sep == line.strip():
            sep_counter += 1
            if ((sep_counter % 2) != 0) and (sep_counter != 1):
                gpr = GeoPlotReader(header_curr)
                gpr.plot(data=data_curr)
                header_curr = []
                data_curr = []
        else:
            if (sep_counter % 2) == 0:
                data_curr.append(line.strip())
            else:
                header_curr.append(line.strip())

    if header_curr or data_curr:
        gpr = GeoPlotReader(header_curr)
        gpr.plot(data=data_curr)
        gpr.save()
        #gpr.show()