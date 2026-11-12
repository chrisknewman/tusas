#!/usr/bin/env python


from os import listdir
from os.path import isfile, join

import argparse as ap
import matplotlib.pyplot as plt
import numpy as np


def get_rms(filename):
    with open(filename, 'r') as file:
        for line in file: pass
        last_line = line
    # END with
    return float(last_line.split()[-1])
# END read_rms()

def get_field(field, filename):
    if field == 'dx':
        search = 'mesh'
    elif field == 'cab' or field == 'bc':
        search = 'case'
    else:
        search = field
    # END if

    entry = list(filter(lambda x: search in x, filename.split('_')))[0]
    val = entry.split('@')[-1]

    if field == 'dx':
        nelems = int(val.split('x')[-1])
        val = 100 / nelems
    elif field == 'cab':
        val = 'constcab' if 'constcab' in val else 'varicab'
    elif field == 'bc':
        val = 'dirichlet' if 'dirichlet' in val else 'neumann'
    elif field == 'dt':
        val = float(val)
    # END if

    if type(val) is str and '.dat' in val:
        val = val.replace('.dat', '')
    # END if

    return val
# END get_field()

def get_data(files):
    return {'dt': [get_field('dt', file) for file in files],
            'dx': [get_field('dx', file) for file in files],
            'theta': [get_field('theta', file) for file in files],
            'case': [get_field('case', file) for file in files],
            'var': [get_field('case', file) for file in files],
            'rms': [get_rms(file) for file in files],}
# END get_data()

def get_files(file_list, case, var, theta, mesh):
    #print(case, var, cab, bc, theta, mesh)


    files = []
    for file in file_list:
        match = get_field('case', file) == case \
                and get_field('var', file) == var \
                and get_field('theta', file) == theta \
                and get_field('mesh', file) == mesh
        #print(match)
        if match: files.append(file)
    # END for

    return sorted(files, key=lambda x: get_field('dt', x), reverse=True)
# END get_files()

def get_rms(filename):
    with open(filename, 'r') as file:
        for line in file: pass
        last_line = line
    # END with
    return float(last_line.split()[-1])
# END read_rms()

def get_field(field, filename):
    if field == 'dx':
        search = 'mesh'
    elif field == 'cab' or field == 'bc':
        search = 'case'
    else:
        search = field
    # END if

    entry = list(filter(lambda x: search in x, filename.split('_')))[0]
    val = entry.split('@')[-1]

    if field == 'dx':
        nelems = int(val.split('x')[-1])
        val = 100 / nelems
    elif field == 'cab':
        val = 'constcab' if 'constcab' in val else 'varicab'
    elif field == 'bc':
        val = 'dirichlet' if 'dirichlet' in val else 'neumann'
    elif field == 'dt':
        val = float(val)
    # END if

    if type(val) is str and '.dat' in val:
        val = val.replace('.dat', '')
    # END if

    return val
# END get_field()

def get_data(files):
    return {'dt': [get_field('dt', file) for file in files],
            'dx': [get_field('dx', file) for file in files],
            'theta': [get_field('theta', file) for file in files],
            'case': [get_field('case', file) for file in files],
            'var': [get_field('case', file) for file in files],
            'rms': [get_rms(file) for file in files],}
# END get_data()

def get_files(file_list, case, var, theta, mesh):
    #print(case, var, cab, bc, theta, mesh)


    files = []
    for file in file_list:
        match = get_field('case', file) == case \
                and get_field('var', file) == var \
                and get_field('theta', file) == theta \
                and get_field('mesh', file) == mesh
        #print(match)
        if match: files.append(file)
    # END for

    return sorted(files, key=lambda x: get_field('dt', x), reverse=True)
# END get_files()

def do_plot(err_type, case, var, theta, meta):
    cab = case.split('-')[2]
    bcs = case.split('-')[3]
    file_list = [f for f in listdir(meta['datadir']) if isfile(join(meta['datadir'], f)) and err_type in f]
    
    perf_plotted = False
    fig_created = False
    for mesh_str in meta['mesh_strs']:
        files = get_files(file_list, case, var, theta, mesh_str)
        files = [join(meta['datadir'], file) for file in files]
        if not files: continue
        if not fig_created:
            fig, ax = plt.subplots()
            fig_created = True
        # END if
        data = get_data(files)

        if not perf_plotted:
            perfect_rate = meta['perfect_rates'][str(data['theta'][0])]
            perfect_scale = data['rms'][0] / (data['dt'][0]**perfect_rate)
            perfect = perfect_scale * np.asarray(data['dt'])**perfect_rate
            
            ax.loglog(data['dt'], perfect, 'k--', label='order ' + str(perfect_rate))
            perf_plotted = True
        # END if 
        
        ax.loglog(data['dt'], data['rms'], 'x--', label='dx = ' + f'{data['dx'][0]:0.6g}') 
    # END for
    if not fig_created: return
    
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    ax.grid(which='minor', linestyle='--', alpha=0.5)
    ax.grid(which='major', linestyle='--', alpha=0.5)

    title = f'{meta['names'][case]}\n' \
            + f'Temporal Convergence of {meta['names'][var]}\n' \
            + f'{meta['names'][cab]}, {bcs.capitalize()} BCs, $\\theta = {theta}$'
    ax.set(xlabel='$\\Delta t$',
           ylabel=meta['names'][err_type],
           title=title)

    if meta['fixed_ranges']: ax.set(ylim=meta['ranges'][var])
    if meta['fixed_ticks']:
        ax.set(xticks=meta['time_ticks'])
        ax.set(xticklabels=meta['time_labels'])
    # END if



    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax.get_xminorticklabels(), rotation=30, horizontalalignment='right')

    fig.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncols=2)
    fig.tight_layout()
    
    plt.savefig(f'{meta['outdir']}/err@{err_type.lower()}_case@{case}_theta@{theta.lower()}_var@{var}.{meta['img_ext']}', bbox_inches='tight')
    
    return
# END do_plot()

def main(args):
    meta = {'datadir': args.input_dir,
            'outdir': args.output_dir,
            'img_ext': args.file_type,
            'perfect_rates': {'0.5': 2,
                              '1.0': 1},
            'mesh_strs': ['1x250', '1x500', '1x1000', '1x2000'],
            'names': {'mms-eta-constcab-dirichlet': 'AC Equation Only',
                      'mms-eta-constcab-neumann': 'AC Equation Only',
                      'mms-coupled-constcab-dirichlet': 'Coupled AC-CH',
                      'mms-coupled-constcab-neumann': 'Coupled AC-CH',
                      'constcab': 'Constant $c_a$ and $c_b$',
                      'eta': '$\\eta$',
                      'c': '$c$',
                      'mu': '$\\mu$',
                      'RMS': 'RMS Error',
                      'DIFF': 'DIFF Error'},
            'fixed_ranges': True,
            'ranges': {'eta': [1e-8, 1e-1],
                       'mu': [1e-8, 1e-1],
                       'c': [1e-8, 1e-1]},
            'fixed_ticks': True,
            'time_ticks': [2e-3, 1e-2, 1e-1],
            'time_labels': ['$2 \\times 10^{-3}$', '$10^{-2}$', '$10^{-1}$']}

    err_types = ['RMS', 'DIFF']
    base_names = ['mms-eta', 'mms-coupled']
    cabs = ['constcab']
    bcs = ['neumann']
    thetas = ['1.0', '0.5']
    variables = ['eta', 'c']


    for err_type in err_types:
        for base in base_names:
            for cab in cabs:
                for bc in bcs:
                    for theta in thetas:
                        casename = base + '-' + cab + '-' + bc
                        for var in variables:
                            do_plot(err_type, casename, var, theta, meta)
                        # END for
                    # END for 
                # END for
            # END for
        # END for
    # END for

    return
# END main()


if __name__ == '__main__':
    parser = ap.ArgumentParser()

    parser.add_argument(
        '-i', '--input-dir',
        dest='input_dir',
        type=str,
        required=True,
        help='Path to input directory.',
    )
    parser.add_argument(
        '-o', '--output-dir',
        dest='output_dir',
        type=str,
        required=False,
        default='./plots',
        help='Directory where output plots will be placed.',
    )
    parser.add_argument(
        '-t', '--file-type',
        dest='file_type',
        type=str,
        required=False,
        default='png',
        help='File type for output images.',
    )
    
    args = parser.parse_args()
    main(args)
# END if

