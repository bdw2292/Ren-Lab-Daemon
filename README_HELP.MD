## Help

## Usage
* bashrcpath is optional argument, if not given the bashrcpaths are extracted from the text file tinkerbashrcpaths.txt. This textfile contains columns for linux OS, CUDA version and CPU / GPU tinker bashrc envioronments. The example CPU envioronment also includes poltype envioronment settings for running QM jobs.
* jobinfofilepath is required argument
* cpunodesonly and gpunodesonly are optional arguments to reduce time pinging nodes for information in cluster 

```
python path_to_submit.py --jobinfofilepath=jobinfofilepath
python path_to_submit.py --jobinfofilepath=jobinfofilepath --cpunodesonly
python path_to_submit.py --jobinfofilepath=jobinfofilepath --gpunodesonly
python path_to_submit.py --bashrcpath=bashrcpath --jobinfofilepath=jobinfofilepath

```

## Job Info File Path
* jobinfofilepath is a text file with possible formats below
* scrachdir and scratchspace are optional assignments for QM jobs
* outputlogpath outputs status of job from daemon to outputlog file 
* if jobpath is not specified, default path is outputlogpath directory

```
--job=command --outputlogpath=log --scratchdir=scratchdir --scratchspace=disk --jobpath=path_to_job
--job=command --outputlogpath=log --jobpath=path_to_job
--job=command --outputlogpath=log
```

* Example file contents for jobinfofilepath
```
--job=cd /home/bdw2292/PoltypeJobs/NewMethanolTest/methanol_test/qm-torsion && psi4 methanol-opt-1_1-2-254.psi4 methanol-opt-1_1-2-254.log --outputlogpath=/home/bdw2292/PoltypeJobs/NewMethanolTest/methanol_test/methanol-poltype.log --scratchdir=/scratch/bdw2292/Psi4-methanol --scratchspace=65GB
--job=cd /home/bdw2292/PoltypeJobs/NewMethanolTest/methanol_test/qm-torsion && psi4 methanol-opt-1_1-2-284.psi4 methanol-opt-1_1-2-284.log --outputlogpath=/home/bdw2292/PoltypeJobs/NewMethanolTest/methanol_test/methanol-poltype.log --scratchdir=/scratch/bdw2292/Psi4-methanol --scratchspace=65GB
```

* Example with dynamic_omm.x (needs to be in $PATH defined in bashrc files)
```
--job=dynamic_omm.x /home/bdw2292/Sims/Aniline/solvwaterboxmin.xyz -k /home/bdw2292/Sims/Aniline/aniline.key 16666 3 100 2 30 N > /home/bdw2292/Sims/Aniline/Aniline_30_16666.out --outputlogpath=/home/bdw2292/Sims/Aniline/Aniline_30_16666.out
```
* Example with jobpath= (changes directory to jobpath)
```
--job=dynamic_omm.x solvwaterboxproddyn.xyz -k aniline_lambda.key 166666 3 2 2 298 N > /home/bdw2292/Sims/Aniline/SolvSim/SolvSimEle1_Vdw1/SolvSimEle1_Vdw1.out --outputlogpath=/home/bdw2292/Sims/Aniline/proddynamicsjobs.txt --jobpath=/home/bdw2292/Sims/Aniline/SolvSim/SolvSimEle1_Vdw1
--job=dynamic_omm.x solvwaterboxproddyn.xyz -k aniline_lambda.key 166666 3 2 2 298 N > /home/bdw2292/Sims/Aniline/SolvSim/SolvSimEle.5_Vdw1/SolvSimEle.5_Vdw1.out --outputlogpath=/home/bdw2292/Sims/Aniline/proddynamicsjobs.txt --jobpath=/home/bdw2292/Sims/Aniline/SolvSim/SolvSimEle.5_Vdw1
```



## Hostname Node Topology
* Saved in nodes.txt
* include a \# in the line to ignore hostnames


## How it works
* First all hostnames are pinged to obtain linux OS version and detect GPU cards and the CUDA version. If hostname is unreachable or RSA host keys are missing, the node is just determined as dead and then removed from available node list. If timeout > 5 seconds command does not return anything, then node is assumed unreachable.
* Next all cpunodes are checked again for any program in cpuprogram exception list. Nodes running these programs are removed from available nodes list.
* Then all gpucards are checked for any program in gpu program exception list. Nodes running these programs are removed from available gpu card list.
* Then jobs are assigned evenly to available CPU nodes/ GPU cards. If there are more jobs than nodes, multiple jobs are assigned per available node, however only one job may be active on an available node at any given time. If scratch space is designated, then before assignment, the node is checked for available scratchspace and will not assign to that node if there is not enough scratchspace for a given job.
* Also, there is a program number exception list that is hardcoded to prevent more than a total number of this program being executed across the entire cluster at any given time to prevent network bottlenecks.
* A while loop continuously checks each nodes list of assigned jobs and checks if the subprocess has exited on that node, then resubmits new jobs until the total number of jobs have been finished. 
* Multiple instances of the dameon are not allowed to run simultaneously. If the program is called while an instance is already running, the input jobs are just added to the existing job queue and then the second instance will exit.
 
## CPU Program Exception List
* psi4, g09, g16, cp2k.ssmp, mpirun_qchem, dynamic.x

## GPU Program Exception List
* dynamic_omm.x, dynamic.mixed

## Restricted Total Program Number
* bar.x:10, bar_omm.x:10

## External Queue File
* jobtoinfo.txt is external representation of queue for daemon
* jobtoinfo_TEMP.txt is used when second instance of daemon is submitting to daemon while an instance is already running, this file is detected by main dameon instance and then added to the internal and external queue.

## Logger File
* Daemon outputs important messages to logger file such as, pinging nodes, removing available nodes due to program exceptions, lack of scratch space, submitting commands to nodes, when jobs finish, removing from queue.., when all jobs finish
