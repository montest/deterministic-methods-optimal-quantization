import os
import csv

import imageio

from bokeh.io import export_svg, export_png
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis
from bokeh.plotting import figure, show

mfclvq_color = Viridis[3][1]
lloyd_color = Viridis[3][2]
nr_color = Viridis[3][0]
general_font_size = '14pt'

def get_distortion_from_file(file_path: str):
    with open(file_path, "r") as f_distortion:
        distortion_per_step = csv.reader(f_distortion, delimiter=';')
        to_return = []
        for row in distortion_per_step:
            to_return.append(row)
    return to_return[1:], len(to_return)


def make_plot_distortion(file_path: str, plot, line_color, fill_color, should_save, nbr_step_max_to_print):
    distortion_per_step, number_row = get_distortion_from_file(file_path)
    file_name = file_path.split('/')[-1]
    distri_name = file_name.split('_')[0]
    method_name = file_name.split('_')[1]
    step = list()
    disto = list()
    source = ColumnDataSource(data=dict(step=step, disto=disto))
    plot.circle(x='step', y='disto', source=source, fill_color=fill_color, line_color=line_color, legend_label=method_name)
    plot.line(x='step', y='disto', source=source, line_color=line_color, legend_label=method_name)
    for i, row in enumerate(distortion_per_step):
        print(f"{method_name}: {i+1}/{number_row-1}")
        if i >= nbr_step_max_to_print:
            break
        step.append(int(row[0]))
        disto.append(float(row[1]))
        if should_save:
            dir_path = os.path.join('plots', distri_name, method_name)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
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


def prepare_and_make_gif_normal_distrib():
    plot = figure(plot_width=900, plot_height=500)

    plot.xaxis.axis_label = "Number of iterations"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Distortion"
    plot.yaxis.axis_label_text_font_size = general_font_size

    plot.legend.text_font_size = general_font_size

    plot = make_plot_distortion('distortions/normal_mfclvq_10.txt', plot, fill_color=None, line_color=mfclvq_color,
                                should_save=True, nbr_step_max_to_print=50)
    plot = make_plot_distortion('distortions/normal_lloyd_10.txt', plot, fill_color=None, line_color=lloyd_color,
                                should_save=True, nbr_step_max_to_print=50)
    plot = make_plot_distortion('distortions/normal_nr_10.txt', plot, fill_color=nr_color, line_color=nr_color, should_save=True,
                                nbr_step_max_to_print=50)
    make_gif(['plots/normal/mfclvq', 'plots/normal/lloyd', 'plots/normal/nr'], output_dir='plots/normal')


def prepare_and_make_gif_normal_distrib_v2():
    dir_path = os.path.join('plots', 'normal', 'all')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    plot = figure(plot_width=900, plot_height=500)

    plot.xaxis.axis_label = "Number of iterations"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Distortion"
    plot.yaxis.axis_label_text_font_size = general_font_size

    distortion_per_step_mfclvq, number_row_mfclvq = get_distortion_from_file('distortions/normal_mfclvq_10.txt')
    distortion_per_step_lloyd, number_row_lloyd = get_distortion_from_file('distortions/normal_lloyd_10.txt')
    distortion_per_step_nr, number_row_nr = get_distortion_from_file('distortions/normal_nr_10.txt')

    step_mfclvq = list()
    disto_mfclvq = list()
    step_lloyd = list()
    disto_lloyd = list()
    step_nr = list()
    disto_nr = list()

    source = ColumnDataSource(data=dict(
        step_mfclvq=step_mfclvq, disto_mfclvq=disto_mfclvq,
        step_lloyd=step_lloyd, disto_lloyd=disto_lloyd,
        step_nr=step_nr, disto_nr=disto_nr
    ))

    plot.circle(x='step_mfclvq', y='disto_mfclvq', source=source, fill_color=None, line_color=mfclvq_color, legend_label='mfclvq')
    plot.line(x='step_mfclvq', y='disto_mfclvq', source=source, line_color=mfclvq_color, legend_label='mfclvq')

    plot.circle(x='step_lloyd', y='disto_lloyd', source=source, fill_color=None, line_color=lloyd_color, legend_label='lloyd')
    plot.line(x='step_lloyd', y='disto_lloyd', source=source, line_color=lloyd_color, legend_label='lloyd')

    plot.circle(x='step_nr', y='disto_nr', source=source, fill_color=nr_color, line_color=nr_color, legend_label='disto_nr')
    plot.line(x='step_nr', y='disto_nr', source=source, line_color=nr_color, legend_label='disto_nr')

    for i, (row_mfclvq, row_lloyd, row_nr) in enumerate(zip(distortion_per_step_mfclvq, distortion_per_step_lloyd, distortion_per_step_nr)):
        if i >= 75:
            break
        step_mfclvq.append(int(row_mfclvq[0]))
        disto_mfclvq.append(float(row_mfclvq[1]))

        step_lloyd.append(float(row_lloyd[0]))
        disto_lloyd.append(float(row_lloyd[1]))

        step_nr.append(float(row_nr[0]))
        disto_nr.append(float(row_nr[1]))


        export_png(plot, filename=os.path.join(dir_path, f"{step_mfclvq[-1]}.png"))

    make_gif([dir_path], output_dir=dir_path)


def save_graph_comparison_convergence_methods(title, distrib, N, methods=['mfclvq', 'lloyd', 'nr'], format='png'):
    plot = figure(plot_width=900, plot_height=500)

    plot.title = title
    plot.title.text_font_size = general_font_size

    plot.xaxis.axis_label = "Number of iterations"
    plot.xaxis.axis_label_text_font_size = general_font_size

    plot.yaxis.axis_label = "Distortion"
    plot.yaxis.axis_label_text_font_size = general_font_size

    plot.legend.text_font_size = general_font_size

    if 'mfclvq' in methods:
        plot = make_plot_distortion(f'distortions/{distrib}_mfclvq_{N}.txt', plot, fill_color=None, line_color=mfclvq_color,
                                should_save=False, nbr_step_max_to_print=100)
    if 'lloyd' in methods:
        plot = make_plot_distortion(f'distortions/{distrib}_lloyd_{N}.txt', plot, fill_color=None, line_color=lloyd_color,
                                should_save=False, nbr_step_max_to_print=100)
    if 'nr' in methods:
        plot = make_plot_distortion(f'distortions/{distrib}_nr_{N}.txt', plot, fill_color=nr_color, line_color=nr_color,
                                should_save=False, nbr_step_max_to_print=100)
    if format == 'png':
        export_png(plot, filename=os.path.join('plots', f"{distrib}_{N}.png"))
    else:
        export_svg(plot, filename=os.path.join('plots', f"{distrib}_{N}.svg"))


prepare_and_make_gif_normal_distrib_v2()

save_graph_comparison_convergence_methods("Normal distribution quantization with N=10", 'normal', 10, methods=['mfclvq', 'lloyd', 'nr'], format='svg')
save_graph_comparison_convergence_methods("Log-normal distribution quantization with N=10", 'lognormal', 10, methods=['mfclvq', 'lloyd', 'nr'], format='svg')
save_graph_comparison_convergence_methods("Exponential distribution quantization with N=10", 'exponential', 10, methods=['mfclvq', 'lloyd', 'nr'], format='svg')

save_graph_comparison_convergence_methods("Normal distribution quantization with N=50", 'normal', 50, methods=['mfclvq', 'lloyd'], format='svg')
save_graph_comparison_convergence_methods("Log-normal distribution quantization with N=50", 'lognormal', 50, methods=['mfclvq', 'lloyd'], format='svg')
save_graph_comparison_convergence_methods("Exponential distribution quantization with N=50", 'exponential', 50, methods=['mfclvq', 'lloyd'], format='svg')


