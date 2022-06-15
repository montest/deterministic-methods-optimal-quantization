import os
import csv

import imageio

from bokeh.io import export_svg, export_png
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show


def make_plot_distortion(file_path: str, plot, line_color, fill_color, should_save, nbr_step_max_to_print):
    file_name = file_path.split('/')[-1]
    distri_name = file_name.split('_')[0]
    method_name = file_name.split('_')[1]
    dir_path = os.path.join('plots', distri_name, method_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    with open(file_path, "r") as f_distortion:
        number_row = sum(1 for line in f_distortion)
    with open(file_path, "r") as f_distortion:
        distortion_per_step = csv.reader(f_distortion, delimiter=';')
        step = list()
        disto = list()
        source = ColumnDataSource(data=dict(step=step, disto=disto))
        plot.circle(x='step', y='disto', source=source, fill_color=fill_color, line_color=line_color, legend_label=method_name)
        plot.line(x='step', y='disto', source=source, line_color=line_color, legend_label=method_name)
        next(distortion_per_step)
        for i, row in enumerate(distortion_per_step):
            print(f"{method_name}: {i+1}/{number_row-1}")
            if i >= nbr_step_max_to_print:
                break
            step.append(int(row[0]))
            disto.append(float(row[1]))
            if should_save:
                export_png(plot, filename=os.path.join(dir_path, f"{step[-1]}.png"))
        return plot


def extract_step_number_from_filename(filename):
    return int(filename[:filename.rfind(".")])


def make_gif(directories, output_dir):
    images = []
    for directory in directories:
        filenames = [filename for filename in os.listdir(directory) if filename.endswith('.png')]
        filenames.sort(key=extract_step_number_from_filename)

        for filename in filenames:
            images.append(imageio.imread(os.path.join(directory, filename)))

    gif_name = 'distortion_convergence.gif'
    imageio.mimsave(os.path.join(output_dir, gif_name), images)


plot = figure(plot_width=700, plot_height=400)
plot.xaxis.axis_label = "Number of iterations"
plot.yaxis.axis_label = "Distortion"

plot = make_plot_distortion('normal_mfclvq_10.txt', plot, fill_color=None, line_color="coral", should_save=True, nbr_step_max_to_print=50)
plot = make_plot_distortion('normal_lloyd_10.txt', plot, fill_color=None, line_color="olivedrab", should_save=True, nbr_step_max_to_print=50)
plot = make_plot_distortion('normal_nr_10.txt', plot, fill_color="gold", line_color="gold", should_save=True, nbr_step_max_to_print=50)

make_gif(['plots/normal/mfclvq', 'plots/normal/lloyd', 'plots/normal/nr'], output_dir='plots/normal')
