# ======================================================================
# Library:
# ======================================================================

from __future__ import print_function, division

import os
import sys

import argparse
import math

# ======================================================================
# COnst
# ======================================================================
# #SBATCH --exclusive

_GPU_SLURM = \
"""#!/bin/bash
# author: trungnt
#SBATCH --nodes %d
#SBATCH -p %s
#SBATCH -t %02d:%02d:00
#SBATCH --begin=now+%dminute
#SBATCH -J %s
#SBATCH -o %s
#SBATCH -e %s
#SBATCH --mem=%d
#SBATCH --gres=gpu:%d
#SBATCH --mail-type=BEGIN,FAIL,END # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=anonymouswork90@gmail.com # Email to which notifications will be sent
#SBATCH

source $HOME/a7dea06c655dcec82784/modules
source activate ai

# run your script
%s

source deactivate
"""

_CPU_SLURM = \
"""#!/bin/bash
# author: trungnt
#SBATCH --nodes %d
#SBATCH -t %02d:%02d:00
#SBATCH --begin=now+%dminute
#SBATCH -J %s
#SBATCH -o %s
#SBATCH -e %s
#SBATCH --constraint="snb|hsw"
#SBATCH -p %s
#SBATCH --ntasks %d
#SBATCH --mem-per-cpu=%d
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=anonymouswork90@gmail.com

module purge
module load gcc/4.9.1
module load mkl/11.1.1
module load intelmpi/5.0.2
module load git
module load openblas

source activate ai-cpu

# run your script
%s

source deactivate
"""

# ======================================================================
# SLURM creator
# ======================================================================
def _create_slurm_gpu(task_name, duration, delay, command, n_gpu=1, mem=15000):
    '''
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python $HOME/appl_taito/src/nist_lre15/models1.py model3 set03
    script = [[path, params ...], [path, params ...]]
    '''
    n_node = 1
    hour = int(math.floor(duration / 60))
    minute = duration - hour * 60
    log_path = task_name + '.out'
    task_name = task_name
    mem = int(mem)

    # ====== Select partition ====== #
    arch = 'gpu'
    if hour == 0 and minute <= 15:
        arch = 'gputest'

    # ====== Select number of node ====== #
    # n_node = math.ceil(n_gpu / 2)

    # ====== Create multiple script ====== #
    if isinstance(command, str) or not hasattr(command, '__len__'):
        command = [command]
    command = ';'.join(command)

    # SBATCH --exclusive
    slurm_text = _GPU_SLURM % (n_node, arch, hour, minute, delay, task_name, log_path, log_path, mem, n_gpu, command)
    f = open('tmp_train_gpu.slurm', 'w')
    f.write(slurm_text)
    f.close()
    os.system('sbatch tmp_train_gpu.slurm')
    os.system('cat tmp_train_gpu.slurm')
    os.remove('tmp_train_gpu.slurm')
    return slurm_text

def gpu_theano(task_name, duration, script, n_gpu=1, mem=15000, delay=0):
    if isinstance(script, str) or not hasattr(script, '__len__'):
        script = [script]
    running_prefix = 'THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python '
    running_script = ''
    for s in script:
        running_script += running_prefix
        running_script += s # path to script
        running_script += ';'
    running_script = running_script[:-1]
    _create_slurm_gpu(task_name, duration, delay, running_script, n_gpu, mem)

def gpu(task_name, duration, script, n_gpu=1, mem=15000, delay=0):
    if isinstance(script, str) or not hasattr(script, '__len__'):
        script = [script]
    running_script = ''
    for s in script:
        running_script += s
        running_script += ';'
    running_script = running_script[:-1]
    _create_slurm_gpu(task_name, duration, delay, running_script, n_gpu, mem)

def _create_slurm_cpu(task_name, duration, delay, command, nb_core=8, mem=15000):
    '''
    parallel    : 16-672 cores /  5mins/3days
    serial      : 1-24 cores   /  5mins/3days
    longrun     : 1-24 cores   /  5mins/14days
    test        : 1-32 cores   /  5mins/30mins
    hugemem     : 1-32 cores   /  5mins/7days
    '''
    n_node = 1
    hour = int(math.floor(duration / 60))
    minute = duration - hour * 60
    log_path = task_name + '.out'
    task_name = task_name
    nb_core = int(nb_core)

    mem = int(mem / float(nb_core))
    machine_type = 'serial'
    if mem > 16000:
        machine_type = 'hugemem'

    # ====== calculate number of node ====== #
    n_node = math.ceil(nb_core / 24)
    if mem > 2500:
        n_node = max(n_node, math.ceil(mem / 2500))

    # ====== Join multiple command ====== #
    if isinstance(command, str) or not hasattr(command, '__len__'):
        command = [command]
    command = ';'.join(command)

    slurm_text = _CPU_SLURM % (n_node, hour, minute, delay, task_name, log_path, log_path, machine_type, nb_core, mem, command)
    f = open('tmp_train_cpu.slurm', 'w')
    f.write(slurm_text)
    f.close()
    os.system('sbatch tmp_train_cpu.slurm')
    os.system('cat tmp_train_cpu.slurm')
    os.remove('tmp_train_cpu.slurm')
    return slurm_text

def cpu(task_name, duration, script, n_cpu=4, mem=12000, delay=0):
    if isinstance(script, str) or not hasattr(script, '__len__'):
        script = [script]
    running_script = ''
    for s in script:
        running_script += s
        running_script += ';'
    running_script = running_script[:-1]
    _create_slurm_cpu(task_name, duration, delay, running_script, n_cpu, mem)

# ======================================================================
# model
# ======================================================================
def slurm_parser():
    parser = argparse.ArgumentParser(
        description='Science the sh*t out of Deep Learning!',
        version='0.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True)
    # ====== SLURM group ====== #
    group = parser.add_argument_group('SLURM Configuration')

    group.add_argument('-t', action='store', type=str, default='AIgenJOB',
            metavar='str',
            help='Title for SLURM job')
    group.add_argument('-d', action='store', type=int, required=True,
            metavar='int',
            help='Duration of running task in minute')
    group.add_argument('-w', action='store', type=int, default=0,
            metavar='int',
            help='Run the task after minutes, default=0')

    group.add_argument('-p', action='store', choices=('gpu', 'cpu'), default = 'gpu',
            metavar='cpu|gpu',
            help='run the task on GPU or CPU, default=gpu')
    group.add_argument('-m', action='store', type=int, default = 12000,
            metavar='int',
            help='memory for the job, default=12000 MB')
    group.add_argument('-np', action='store', type=int, default = 1,
            metavar='int',
            help='Number of core for CPU task')

    group.add_argument('-f', action='append', required=True, nargs='+',
            metavar='list',
            help='path to python script & its arguments (e.g. script.py arg1 arg2 ...)')

    return parser

# python runner.py -t MFCC -d 30 -p cpu -m 16000 -np 8 -f feature_extract.py set01.dat 10 20 10 2500 8
if __name__ == '__main__':
    parser = slurm_parser()
    if len(sys.argv) > 1:

        results = parser.parse_args()

        if results.p == 'gpu':
            s = _create_slurm_gpu(results.t, results.d, results.w, results.f, results.np, results.m)
            print(s)
        elif results.p == 'cpu':
            s = _create_slurm_cpu(results.t, results.d, results.w, results.f, results.np, results.m)
            print(s)
        else:
            parser.print_help()
    else:
        parser.print_help()


# python runner.py -t 3c52020mfcc  -d 4320 -m 15000 -f model_runner.py m3c 5_20_20_mfcc_set01
# python runner.py -t 3c52010mfcc  -d 4320 -m 15000 -f model_runner.py m3c 5_20_10_mfcc_set01
# python runner.py -t 3c52020attr  -d 4320 -m 15000 -f model_runner.py m3c 5_20_20_attr_set01
# python runner.py -t 2b52010attr  -d 4320 -m 15000 -f model_runner.py m2b 5_20_10_attr_set01

# python runner.py -t 2e52020attr  -d 4320 -m 15000 -f model_runner.py m2e 5_20_20_attr_set01 train.yaml
# python runner.py -t 2e52010attr  -d 4320 -m 15000 -f model_runner.py m2e 5_20_10_attr_set01 train.yaml
# python runner.py -t 2e52020mfcc  -d 4320 -m 15000 -f model_runner.py m2e 5_20_20_mfcc_set01 train.yaml

# python runner.py -t 2f52020attr  -d 4320 -m 15000 -f model_runner.py m2f 5_20_20_attr_set01 train.yaml
# python runner.py -t 2f52010attr  -d 4320 -m 15000 -f model_runner.py m2f 5_20_10_attr_set01 train.yaml

# python runner.py -t 2d52020attr  -d 4320 -m 15000 -f model_runner.py m2d 5_20_20_attr_set01 train.yaml
# python runner.py -t 2d52010attr  -d 4320 -m 15000 -f model_runner.py m2d 5_20_10_attr_set01 train.yaml
# python runner.py -t 2d52020mfcc  -d 4320 -m 15000 -f model_runner.py m2d 5_20_20_mfcc_set01 train.yaml
# python runner.py -t 2d52010mfcc  -d 4320 -m 15000 -f model_runner.py m2d 5_20_10_mfcc_set01 train.yaml

# python runner.py -t 2c52020attr  -d 4320 -m 15000 -f model_runner.py m2c 5_20_20_attr_set01 train.yaml
# python runner.py -t 2c52010attr  -d 4320 -m 15000 -f model_runner.py m2c 5_20_10_attr_set01 train.yaml
# python runner.py -t 2c52020mfcc  -d 4320 -m 15000 -f model_runner.py m2c 5_20_20_mfcc_set01 train.yaml
# python runner.py -t 2c52010mfcc  -d 4320 -m 15000 -f model_runner.py m2c 5_20_10_mfcc_set01 train.yaml

# python runner.py -t 2b52020mfcc  -d 4320 -m 15000 -f model_runner.py m2b 5_20_20_mfcc_set01 train.yaml
# python runner.py -t 2b52010mfcc  -d 4320 -m 15000 -f model_runner.py m2b 5_20_10_mfcc_set01 train.yaml
# python runner.py -t 2b52010attr  -d 4320 -m 15000 -f model_runner.py m2b 5_20_10_attr_set01 train.yaml
