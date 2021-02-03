import os
import sys
import traceback
import subprocess
import time
import getopt
import re
from itertools import islice
from tqdm import tqdm

global nodelistfilepath
global inputbashrcpath
global restrictedprogramtonumber
global currentrestrictedprogramtoprocesslist
global pidfile
global loggerfile
global errorloggerfile
global jobtoinfo
global masterloghandle
global writepath
global cpuprogramexceptionlist
global cpuprogramlist
global gpuprogramlist
global bashrcfilename
global cpunodesonly
global gpunodesonly
global nodetimeout
global verbosemode
global mastererrorloghandle

curdir = os.path.dirname(os.path.realpath(__file__))+r'/'
bashrcfilename=curdir+'tinkerbashrcpaths.txt'
nodelistfilepath=curdir+'nodes.txt'
inputbashrcpath=None
jobinfofilepath=None
pidfile=curdir+'daemon.pid' # just in case want to kill daemon
loggerfile=curdir+'logger.txt'
errorloggerfile=curdir+'errorlogger.txt'
jobtoinfo=curdir+'jobtoinfo.txt'
writepath=curdir

masterloghandle=open(loggerfile,'w',buffering=1)
mastererrorloghandle=open(errorloggerfile,'a',buffering=1)
sleeptime=60
cpuprogramexceptionlist=['psi4','g09','g16',"cp2k.ssmp","mpirun_qchem","dynamic.x"] # dont run on cpu node if detect these programs
restrictedprogramtonumber={'bar.x':10,'bar_omm.x':10} # restrictions on program use for network slow downs
currentrestrictedprogramtoprocesslist={'bar.x':[],'bar_omm.x':[]}
cpuprogramlist=['psi4','g09','g16',"cp2k.ssmp","mpirun_qchem","dynamic.x",'minimize.x','minimize','poltype.py'] # run on cpu node env if see these
gpuprogramlist=['dynamic_omm.x','bar_omm.x','dynamic.gpu' , 'dynamic.mixed' , 'analyze.gpu' , 'analyze.mixed' , 'bar.gpu', 'bar.mixed'] # run on gpu node env if see these
cpunodesonly=False
gpunodesonly=False
nodetimeout=5 # nodetime out if checking node features with command stalls
verbosemode=False

opts, xargs = getopt.getopt(sys.argv[1:],'',["bashrcpath=","jobinfofilepath=","cpunodesonly","gpunodesonly","verbosemode"])
for o, a in opts:
    if o in ("--bashrcpath"):
        inputbashrcpath=a
    elif o in ("--jobinfofilepath"):
        jobinfofilepath=a
    elif o in ("--cpunodesonly"):
        cpunodesonly=True
    elif o in ("--gpunodesonly"):
        gpunodesonly=True
    elif o in ("--verbosemode"):
        verbosemode=True


def RemoveAlreadyActiveNodes(nodelist,programexceptionlist=None,gpunodes=False):
    newnodelist=[]
    if verbosemode==True:
        disablebool=False
    else:
        disablebool=True

    for nodeidx in tqdm(range(len(nodelist)),desc='Checking already active nodes',disable=disablebool):
        node=nodelist[nodeidx]
        keepnode=True
        nonactivecards=[]
        if gpunodes==True:
            nonactivecards=CheckWhichGPUCardsActive(node)
        else:
            exceptionstring=''
            for exception in programexceptionlist:
                exceptionstring+=exception+'|'
            exceptionstring=exceptionstring[:-1]
            cmdstr='pgrep '+exceptionstring
            output=CheckOutputFromExternalNode(node,cmdstr)
            if output==False:
                pass
            else:
                keepnode=False
                cmdstr1='ps -p %s'%(output)
                cmdstr2='ps -p %s'%(output)+' -u'
                output1=CheckOutputFromExternalNode(node,cmdstr1)
                output2=CheckOutputFromExternalNode(node,cmdstr2)
                if type(output1)==str:
                    WriteToLogFile(output1)
                if type(output2)==str:
                    WriteToLogFile(output2)

        if keepnode==True:
            if gpunodes==True:
                for card in nonactivecards:
                    newnodelist.append(node+'-'+str(card))

            else:
                newnodelist.append(node)
        else:
            WriteToLogFile('cannot use gpunode '+node+' because of program exception')

    return newnodelist

def PingNodesAndDetermineNodeInfo(nodelist):
    cpunodes=[]
    gpunodes=[] # includes the -0, -1 after for cards etc...
    nodetoosversion={}
    gpunodetocudaversion={}
    if verbosemode==True:
        disablebool=False
    else:
        disablebool=True
    for nodeidx in tqdm(range(len(nodelist)),desc='Pinging nodes',disable=disablebool):
        node=nodelist[nodeidx]
        osversion,nodedead=CheckOSVersion(node)
        if nodedead==False:
            nodetoosversion[node]=osversion
            cudaversion,cardcount=CheckGPUStats(node)
            
            if cudaversion!=None:
                for i in range(cardcount+1):
                    newnode=node+'-'+str(i)
                    gpunodetocudaversion[newnode]=cudaversion
                    gpunodes.append(newnode)
            cpunodes.append(node)
        else:
            continue


    return cpunodes,gpunodes,nodetoosversion,gpunodetocudaversion

           
def CheckOSVersion(node):
    osversion=None
    cmdstr='cat /etc/system-release'
    job='ssh %s "%s"'%(node,cmdstr)
    p = subprocess.Popen(job, stdout=subprocess.PIPE,shell=True)
    nodedead,output=CheckForDeadNode(p,node)

    if nodedead==False:
        found=False
        linesplit=output.split()
        for eidx in range(len(linesplit)):
            e=linesplit[eidx]
            if 'release' in e:
                specialidx=eidx+1
                found=True
            if found==True:
                if eidx==specialidx:
                    version=e
        if found==False:
            nodedead=True
        else:
            versionsplit=version.split('.')
            osversion=versionsplit[0]
    return osversion,nodedead


def CheckGPUStats(node):
    cmdstr='nvidia-smi'
    job='ssh %s "%s"'%(node,cmdstr)
    if cpunodesonly==False:
        p = subprocess.Popen(job, stdout=subprocess.PIPE,shell=True)
        nodedead,output=CheckForDeadNode(p,node)
    cudaversion=None
    cardcount=-1
    nodedead=False
    if nodedead==False and cpunodesonly==False:
        cardcount=0
        lines=output.split('\n')
        for line in lines:
            linesplit=line.split()
            if 'CUDA' in line:
                cudaversion=linesplit[8]
            if len(linesplit)==15:
                cardcount+=1
    return cudaversion,cardcount

def CheckWhichGPUCardsActive(node):
    nonactivecards=[]
    cmdstr='nvidia-smi'
    job='ssh %s "%s"'%(node,cmdstr)
    p = subprocess.Popen(job, stdout=subprocess.PIPE,shell=True)
    nodedead,output=CheckForDeadNode(p,node)
    lines=output.split('\n')
    count=-1
    for line in lines:
        linesplit=line.split()
        if len(linesplit)==15:
            percent=linesplit[12]
            value=percent.replace('%','')
            if value.isnumeric():
                value=float(value)
                count+=1
                if value<10:
                    nonactivecards.append(count)
                else:
                    card=node+'-'+str(count)
                    WriteToLogFile('GPU card '+card+' is currently active and cannot use')
            else:
                card=node+'-'+str(count)
                WriteToLogFile('GPU card '+card+' has problem in nvidia-smi and cannot use')


    return nonactivecards

def CheckOutputFromExternalNode(node,cmdstr):
    output=True
    if node[-1].isdigit() and '-' in node:
        node=node[:-2]
    job='ssh %s "%s"'%(node,cmdstr)

    try: # if it has output that means this process is running
         output=subprocess.check_output(job,stderr=subprocess.STDOUT,shell=True,timeout=nodetimeout)
         output=ConvertOutput(output)
    except: #if it fails, that means no process with the program is running or node is dead/unreachable
         output=False
    return output 

def ReadNodeList(nodelistfilepath):
    nodelist=[]
    gpunodesonlylist=[]
    cpunodesonlylist=[]
    if os.path.isfile(nodelistfilepath):
        temp=open(nodelistfilepath,'r')
        results=temp.readlines()
        for line in results:

            newline=line.replace('\n','')
            if '#' not in line:
                linesplit=newline.split()
                node=linesplit[0]
                nodelist.append(node)
                if 'CPUONLY' in line:
                    cpunodesonlylist.append(node)
                elif 'GPUONLY' in line:
                    gpunodesonlylist.append(node)

            else:
                WriteToLogFile('Removing from node list '+newline)    

        temp.close()
    return nodelist,cpunodesonlylist,gpunodesonlylist

def chunks(lst, n):
    ls=[]
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        ls.append(lst[i:i + n])
    return ls

def JobsPerNodeList(jobs,nodes):
    if len(nodes)==0:
        jobspernodelist=[]
    else:
        if len(jobs)>len(nodes):
            remainder=len(jobs) % len(nodes)
            wholejobs=len(jobs)-remainder
            jobspernode=int(wholejobs/len(nodes))
        else:
            jobspernode=1
            wholejobs=len(jobs)
        jobspernodelist=chunks(jobs[:wholejobs],jobspernode)
    return jobspernodelist


def CheckScratchSpaceAllJobs(jobspernodelist,nodes,jobtoscratchspace):
    newnodes=[]
    for i in range(len(jobspernodelist)):
       joblist=jobspernodelist[i]
       node=nodes[i]
       isitenough=False
       for job in joblist:
           scratchspaceneeded=jobtoscratchspace[job]
           if scratchspaceneeded!=None:
               scratchavail=CheckScratchSpace(node)
               if scratchavail!=False:
                   isitenough=CheckIfEnoughScratch(scratchspaceneeded,scratchavail)
           else:
               isitenough=True
           if isitenough==False:
               WriteToLogFile('not enough scratch space for job = '+job+' on node '+node)
           else:
               newnodes.append(node)
    return newnodes


def CheckBashrcPathsAllJobs(nodes,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths):
    newnodes=[]
    for node in nodes:
        bashrcpath,accept=DetermineBashrcPath(nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,node)
        if accept==True:
            newnodes.append(node)
    return newnodes

def CheckRAM(node):
    ram=False
    cmdstr='free -g'
    output=CheckOutputFromExternalNode(node,cmdstr)
    if output!=False:
        lines=output.split('\n')
        for line in lines:
            linesplit=line.split()
            if 'Mem' in line:
                ram=float(linesplit[3])
            elif 'buffers/cache' in line:
                ram=float(linesplit[3]) 

    return ram

def CheckRAMAllJobs(jobspernodelist,nodes,jobtoram):
    newnodes=[]
    for i in range(len(jobspernodelist)):
       joblist=jobspernodelist[i]
       node=nodes[i]
       isitenough=False
       for job in joblist:
           ramneeded=jobtoram[job]
           if ramneeded!=None:
               ramavail=CheckRAM(node)
               if ramavail!=False:
                   isitenough=CheckIfEnoughRAM(ramneeded,ramavail)
           else:
               isitenough=True
           if isitenough==False:
               WriteToLogFile('not enough ram for job = '+job+' on node '+node)
           else:
               newnodes.append(node)
    return newnodes

def CheckIfEnoughRAM(ramneeded,ramavail):
    enough=True
    if float(ramneeded)<ramavail:
        enough=False
    return enough



def DistributeJobsToNodes(nodes,jobs,jobtoscratchspace,prevnodetojoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobtoram):
    nodetojoblist={}
    nodetojoblist.update(prevnodetojoblist)
    if len(nodes)!=0:
        jobspernodelist=JobsPerNodeList(jobs,nodes)
        newnodes=CheckScratchSpaceAllJobs(jobspernodelist,nodes,jobtoscratchspace)
        jobspernodelist=JobsPerNodeList(jobs,newnodes)
        newnodes=CheckBashrcPathsAllJobs(newnodes,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        jobspernodelist=JobsPerNodeList(jobs,newnodes)
        newnodes=CheckRAMAllJobs(jobspernodelist,newnodes,jobtoram)
        jobspernodelist=JobsPerNodeList(jobs,newnodes)
        for i in range(len(jobspernodelist)):
           joblist=jobspernodelist[i]
           node=newnodes[i]
           for job in joblist:
               if node not in nodetojoblist.keys():
                   nodetojoblist[node]=[]
               nodetojoblist[node].append(job)
    for node,joblist in nodetojoblist.items():
        for job in joblist:
            WriteToLogFile('Job '+job+' '+'is assigned to node '+node)
    return nodetojoblist



def CheckScratchSpace(node):
    cmdstr='df -h'
    scratchavail=False
    output=CheckOutputFromExternalNode(node,cmdstr)
    if output!=False:
        lines=output.split('\n')[1:-1]
        d={}
        for line in lines:
            linesplit=line.split()
            if len(linesplit)==5 or len(linesplit)==6:
                avail = re.split('\s+', line)[3]
                mount = re.split('\s+', line)[5]
                d[mount] = avail
        if '/scratch' in d.keys(): 
            scratchavail=d['/scratch']
        else:
            scratchavail='0G'
            WriteToLogFile(' node '+node+' has no scratch')
        if scratchavail==False:
            cmdstr="du -h /scratch | sort -n -r | head -n 15"
            output=CheckOutputFromExternalNode(node,cmdstr)
            WriteToLogFile(output)

    return scratchavail


def CheckIfEnoughScratch(scratchinput,scratchspace):
    enoughscratchspace=False
    if scratchinput==None:
        enoughscratchspace=True
        return enoughscratchspace
    inputspace,inputunit=SplitScratch(scratchinput)
    availspace,availunit=SplitScratch(scratchspace)
    if inputunit=='G' and availunit=='M' or (inputunit=='T' and availunit=='M') or (inputunit=='T' and availunit=='G'):
        pass
    elif inputunit==availunit:
        if float(inputspace)<float(availspace):
            enoughscratchspace=True
    elif  (inputunit=='M' and availunit=='G'):
        inputspace=float(inputspace)*.001
        availspace=float(availspace)
        if float(inputspace)<float(availspace):
            enoughscratchspace=True
    elif  (inputunit=='M' and availunit=='T'):
        inputspace=float(inputspace)*.000001
        availspace=float(availspace)
        if float(inputspace)<float(availspace):
            enoughscratchspace=True
    elif  (inputunit=='G' and availunit=='T'):
        inputspace=float(inputspace)*.001
        availspace=float(availspace)
        if float(inputspace)<float(availspace):
            enoughscratchspace=True
     
       
    return enoughscratchspace

def SplitScratch(string):
    for eidx in range(len(string)):
        e=string[eidx]
        if not e.isdigit() and e!='.':
            index=eidx
            break
    space=string[:index]
    diskunit=string[index]
    return space,diskunit



def CheckIfPreviousJobsFinished(jobtoprocess,previousjobs,finishedjoblist,jobtologhandle,node,polledjobs):
    previousjobsfinished=True
    for job in previousjobs:
        if job not in jobtoprocess.keys() and job not in polledjobs:
            previousjobsfinished=False
        else:
            loghandle=jobtologhandle[job]
            finishedjoblist,term,polledjobs=PollProcess(jobtoprocess,job,finishedjoblist,loghandle,node,polledjobs)
            if term==False:
                previousjobsfinished=False
    return previousjobsfinished,finishedjoblist

def WriteToLogFile(string,loghandle=None):
    now = time.strftime("%c",time.localtime())
    if loghandle!=None:
        loghandle.write(now+' '+string+'\n')
        loghandle.flush()
        os.fsync(loghandle.fileno())
    masterloghandle.write(now+' '+string+'\n')
    masterloghandle.flush()
    os.fsync(masterloghandle.fileno())


def CallSubprocess(node,jobpath,bashrcpath,job,loghandle,wait=False):
    if node[-1].isdigit() and '-' in node:
        cardvalue=node[-1]
        node=node[:-2]
        job=SpecifyGPUCard(cardvalue,job)
    cmdstr = 'ssh %s "cd %s;source %s;%s"' %(str(node),jobpath,bashrcpath,job)
    process = subprocess.Popen(cmdstr, stdout=loghandle,stderr=loghandle,shell=True)
    WriteToLogFile('Calling: '+cmdstr,loghandle)
    nodedead=False
    if wait==True: # grab output from subprocess:
        nodedead,output=CheckForDeadNode(process,node)    
    return process,nodedead

def CheckForDeadNode(process,node):
    nodedead=False
    output, err = process.communicate()
    output=ConvertOutput(output)    
    if process.returncode != 0:
        if err!=None:
            err=ConvertOutput(err)
            WriteToLogFile(err+' '+'on node '+node)
            nodedead=True
    return nodedead,output

def ConvertOutput(output):
    if output!=None:
        output=output.rstrip()
        if type(output)!=str:
            output=output.decode("utf-8")
    return output

def MakeScratch(node,jobpath,bashrcpath,loghandle,scratchdir):
    cmdstr='[ -d "%s" ] && echo "Directory Exists"'%(scratchdir)
    output=CheckOutputFromExternalNode(node,cmdstr)
    if CheckOutputFromExternalNode(node,cmdstr)==False:
        mkstr='mkdir '+scratchdir
        process,nodedead=CallSubprocess(node,jobpath,bashrcpath,mkstr,loghandle)                            


def SubmitJob(node,jobpath,bashrcpath,job,loghandle,jobtoprocess,jobtoscratchdir,jobinfo):
    if job in jobtoscratchdir.keys():
        scratchdir=jobtoscratchdir[job]
        if scratchdir!=None:
            MakeScratch(node,jobpath,bashrcpath,loghandle,scratchdir)
        process,nodedead=CallSubprocess(node,jobpath,bashrcpath,job,loghandle)
        jobtoprocess[job]=process
        RemoveJobInfoFromQueue(jobinfo,jobtoprocess)
    return jobtoprocess

def PollProcess(jobtoprocess,job,finishedjoblist,loghandle,node,polledjobs):
    process=jobtoprocess[job]
    poll=process.poll()
    polledjobs.append(job)
    if poll!=None:
        out, err = process.communicate()
        if job not in finishedjoblist:
            finishedjoblist.append(job)
            WriteToLogFile(job+' '+'has terminated on node '+node,loghandle)
            if process.returncode != 0:
                WriteToLogFile('Error detected for job '+job,loghandle=mastererrorloghandle)
        for program in restrictedprogramtonumber.keys():
            if program in job:
                plist=currentrestrictedprogramtoprocesslist[program]
                if process in plist:
                    currentrestrictedprogramtoprocesslist[program].remove(process)
        term=True
        
    else:
        for program in restrictedprogramtonumber.keys():
            if program in job:
                plist=currentrestrictedprogramtoprocesslist[program]
                if process not in plist:
                    currentrestrictedprogramtoprocesslist[program].append(process) 
        WriteToLogFile(job+' '+'has not terminated on node '+node,loghandle)
        term=False
    return finishedjoblist,term,polledjobs


def SubmitJobs(cpunodetojoblist,gpunodetojoblist,inputbashrcpath,sleeptime,jobtologhandle,jobinfo,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths):
    jobnumber=len(jobtologhandle.keys())
    jobtoprocess={}
    finishedjoblist=[]
    while len(finishedjoblist)!=jobnumber:
        jobinfo,jobtoprocess,finishedjoblist=SubmitJobsLoop(cpunodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        jobinfo,jobtoprocess,finishedjoblist=SubmitJobsLoop(gpunodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)

        time.sleep(sleeptime)
        WriteToLogFile('*************************************')
        cpunodes,gpucards,nodetoosversion,gpunodetocudaversion=GrabCPUGPUNodes()
        jobinfo=ReadTempJobInfoFiles(jobinfo)
        AddJobInfoToDictionary(jobinfo,jobtoinfo,jobtoprocess)
        jobinfo=ReadJobInfoFromFile(jobinfo,jobtoinfo)
        cpujobs,gpujobs=PartitionJobs(jobinfo,cpuprogramlist,gpuprogramlist,jobtoprocess)
        jobtologhandle=CreateNewLogHandles(jobinfo['logname'],jobtologhandle)
        cpunodetojoblist=DistributeJobsToNodes(cpunodes,cpujobs,jobinfo['scratchspace'],cpunodetojoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'])
        gpunodetojoblist=DistributeJobsToNodes(gpucards,gpujobs,jobinfo['scratchspace'],gpunodetojoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'])
        jobnumber=len(jobtologhandle.keys())
    WriteToLogFile('All jobs have finished ')

def SubmitJobsLoop(nodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths):
    polledjobs=[]
    if verbosemode==True:
        disablebool=False
    else:
        disablebool=True

    for nodeidx in tqdm(range(len(list(nodetojoblist.keys()))),desc='Cycling through available nodes for submission',disable=disablebool):
        node=list(nodetojoblist.keys())[nodeidx]
        joblist=nodetojoblist[node]
        for i in tqdm(range(len(joblist)),desc='Submitting jobs on node %s '+node,disable=disablebool):
            job=joblist[i]
            loghandle=jobtologhandle[job]
            logname=jobinfo['logname'][job]
            logpath,tail=os.path.split(logname)
            jobpath=jobinfo['jobpath'][job]
            if jobpath==None:
                path=logpath
            else:
                path=jobpath
            submit=True
            if job not in jobtoprocess.keys():

                previousjobs=joblist[:i]
                previousjobsfinished,finishedjoblist=CheckIfPreviousJobsFinished(jobtoprocess,previousjobs,finishedjoblist,jobtologhandle,node,polledjobs)
                if previousjobsfinished==False:
                    submit=False
                for program,number in restrictedprogramtonumber.items():
                    if program in job:
                        plist=currentrestrictedprogramtoprocesslist[program]
                        currentnumber=len(plist)
                        if currentnumber>=number:
                            submit=False
                if submit==True: # here need to read bashrcpaths if variable not specified
                    if inputbashrcpath==None:
                        bashrcpath,accept=DetermineBashrcPath(nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,node)
                    else:
                        bashrcpath=inputbashrcpath
                    jobtoprocess=SubmitJob(node,path,bashrcpath,job,loghandle,jobtoprocess,jobinfo['scratch'],jobinfo)
            elif job in jobtoprocess.keys() and job not in polledjobs:
                finishedjoblist,term,polledjobs=PollProcess(jobtoprocess,job,finishedjoblist,loghandle,node,polledjobs)
    return jobinfo,jobtoprocess,finishedjoblist

def GPUCardToNode(gpucards,cpunodesonlylist):
    gpucardtonode={}
    for gpucard in gpucards:
        gpunode=gpucard[:-2]
        if gpunode not in cpunodesonlylist:
            gpucardtonode[gpucard]=gpunode
        else:
            WriteToLogFile('removing node '+gpunode+' from gpunode list since its marked as CPUONLY')

    return gpucardtonode

def SpecifyGPUCard(cardvalue,job):
    string='CUDA_DEVICE_ORDER=PCI_BUS_ID '+';'+' export CUDA_VISIBLE_DEVICES='+str(cardvalue)
    job=string+';'+job
    return job


def RemoveJobInfoFromQueue(jobinfo,jobtoprocess):
    newjobinfo=RemoveAlreadySubmittedJobs(jobtoprocess,jobinfo) # just removing submissions from queue
    WriteOutJobInfo(newjobinfo,jobtoinfo,jobtoprocess)

def RemoveAlreadySubmittedJobs(jobtoprocess,jobinfo):
    newjobinfo={}
    for key in jobinfo.keys():
        d=jobinfo[key]
        if key not in newjobinfo.keys():
            newjobinfo[key]={}
        for job in d.keys():
            if job not in jobtoprocess.keys():
                newjobinfo[key][job]=d[job]
                
    return newjobinfo

def WriteOutJobInfo(jobinfo,filepath,jobtoprocess):
    bufsize=1
    if os.path.isfile(filepath):
        os.remove(filepath)
    temp=open(filepath,'w',buffering=bufsize)
    jobtologname=jobinfo['logname']
    jobtoscratch=jobinfo['scratch']
    jobtoscratchspace=jobinfo['scratchspace']
    jobtojobpath=jobinfo['jobpath']
    jobtoram=jobinfo['ram']
    counter=0
    array=['--scratchdir=','--scratchspace=','--jobpath=','--ram=']
    for job,log in jobtologname.items():
        if job in jobtoprocess.keys():
            continue
        counter+=1
        scratch=jobtoscratch[job]
        scratchspace=jobtoscratchspace[job]
        jobpath=jobtojobpath[job]
        ram=jobtoram[job]
        curarray=[scratch,scratchspace,jobpath,ram]
        string='--job='+job+' '+'--outputlogpath='+log+' '
        for i in range(len(array)):
            input=array[i]
            value=curarray[i]
            if value!=None:
                string+=input+value+' '
        string+='\n'
        temp.write(string)
        temp.flush()
        os.fsync(temp.fileno())

    temp.close()
    WriteToLogFile(str(counter)+' jobs are left in queue')


def AddJobInfoToDictionary(jobinfo,filepath,jobtoprocess):
    jobinfoprev=ReadJobInfoFromFile(jobinfo,jobtoinfo)
    for key in jobinfoprev.keys():
        prevd=jobinfoprev[key]
        jobinfo[key].update(prevd)
    WriteOutJobInfo(jobinfo,filepath,jobtoprocess)


def ParseJobInfo(line):
    linesplit=line.split('--')[1:]
    linesplit=[e.rstrip() for e in linesplit]
    job=None
    logname=None
    scratch=None
    scratchspace=None
    jobpath=None
    ram=None
    for line in linesplit:
        if "job=" in line:
            job=line.replace('job=','')
        if "outputlogpath=" in line:
            logname=line.replace('outputlogpath=','')
        if "scratchdir=" in line:
            scratch=line.replace('scratchdir=','')
        if "scratchspace=" in line:
            scratchspace=line.replace('scratchspace=','')
        if "jobpath=" in line:
            jobpath=line.replace('jobpath=','')
        if "ram=" in line:
            jobpath=line.replace('ram=','')


    return job,logname,scratch,scratchspace,jobpath,ram


def ReadTempJobInfoFiles(jobinfo):
    curdir=os.getcwd()
    os.chdir(writepath)
    files=os.listdir()
    dellist=[]
    for f in files:
        if '_TEMP' in f:
            jobinfo=ReadJobInfoFromFile(jobinfo,f)
            dellist.append(f)
    for f in dellist:
        if os.path.isfile(f):
            os.remove(f)
    return jobinfo

   
def CreateNewLogHandles(jobtologname,jobtologhandle):
    createdloghandles=[]
    lognametologhandle={}
    for job,logname in jobtologname.items():
        if job in jobtologhandle.keys():
            continue
        if logname not in createdloghandles:
            createdloghandles.append(logname)
            loghandle=open(logname,'w')
            lognametologhandle[logname]=loghandle
        else:
            loghandle=lognametologhandle[logname]
        jobtologhandle[job]=loghandle
    return jobtologhandle

def ReadJobInfoFromFile(jobinfo,filename):
    if os.path.isfile(filename):
        temp=open(filename,'r')
        results=temp.readlines()
        temp.close()
        for line in results:
            job,log,scratch,scratchspace,jobpath,ram=ParseJobInfo(line)
            if job==None or log==None:
                continue
            jobinfo['logname'][job]=log
            jobinfo['scratch'][job]=scratch
            jobinfo['scratchspace'][job]=scratchspace
            jobinfo['jobpath'][job]=jobpath
            jobinfo['ram'][job]=ram

    return jobinfo



def WritePIDFile():
    pid=str(os.getpid())
    temp=open(pidfile, 'w',buffering=1)
    temp.write(pid+'\n')
    temp.flush()
    os.fsync(temp.fileno())
    temp.close()

def PartitionJobs(jobinfo,cpuprogramlist,gpuprogramlist,jobtoprocess):
    cpujobs=[]
    gpujobs=[]
    d=jobinfo['logname']
    for job in d.keys():
        if job not in jobtoprocess.keys():
            for program in cpuprogramlist:
                if program in job:
                    cpujobs.append(job)
            for program in gpuprogramlist:
                if program in job:
                    gpujobs.append(job)
    return cpujobs,gpujobs

def GrabCPUGPUNodes():
    WriteToLogFile('*************************************')
    WriteToLogFile("Checking available CPU nodes and GPU cards")
    nodes,cpunodesonlylist,gpunodesonlylist=ReadNodeList(nodelistfilepath)
    cpunodes,gpucards,nodetoosversion,gpunodetocudaversion=PingNodesAndDetermineNodeInfo(nodes)
    WriteToLogFile('*************************************')
    WriteToLogFile("Removing active CPU nodes")
    newcpunodes=[]
    for cpunode in cpunodes:
        if cpunode not in gpunodesonlylist:
            newcpunodes.append(cpunode)
        else:
            WriteToLogFile('removing node '+cpunode+' from cpunode list since its marked as GPUONLY')
    if gpunodesonly==False:
        cpunodes=RemoveAlreadyActiveNodes(newcpunodes,cpuprogramexceptionlist)
    else:
        cpunodes=[]
    
    gpucardtonode=GPUCardToNode(gpucards,cpunodesonlylist)
    gpunodes=list(set(gpucardtonode.values()))
    WriteToLogFile('*************************************')
    WriteToLogFile("Removing active GPU cards")
    if cpunodesonly==False:
        gpucards=RemoveAlreadyActiveNodes(gpunodes,programexceptionlist=None,gpunodes=True)
    else:
        gpucards=[]
    for gpucard in gpucards:
        WriteToLogFile('GPU card '+gpucard+' is available for submission')
    for cpunode in cpunodes:
        WriteToLogFile('CPU node '+cpunode+' is avaible for submission')
    return cpunodes,gpucards,nodetoosversion,gpunodetocudaversion

def ReadInBashrcs(bashrcfilename):
    ostocudaversiontobashrcpaths={}
    temp=open(bashrcfilename,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        linesplit=line.split()
        osversion=linesplit[0]
        cudaversion=linesplit[1]
        tinkeropenmmbashrcpath=linesplit[2]
        cputinkerbashrcpath=linesplit[3]
        if osversion not in ostocudaversiontobashrcpaths.keys():
            ostocudaversiontobashrcpaths[osversion]={}
        if cudaversion not in ostocudaversiontobashrcpaths[osversion].keys():
            ostocudaversiontobashrcpaths[osversion][cudaversion]={}
            ostocudaversiontobashrcpaths[osversion][cudaversion]['gputinkerbashrc']=tinkeropenmmbashrcpath
        ostocudaversiontobashrcpaths[osversion]['cputinkerbashrc']=cputinkerbashrcpath
    return ostocudaversiontobashrcpaths 

def DetermineBashrcPath(nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,node):
    usinggpunode=False
    bashrcpath=None
    accept=True
    if node[-1].isdigit() and '-' in node:
        gpunode=node
        node=node[:-2]
        usinggpunode=True
    osversion=nodetoosversion[node]
    dic=ostocudaversiontobashrcpaths[osversion]
    if usinggpunode==True:
        cudaversion=gpunodetocudaversion[gpunode]
        if cudaversion in dic.keys():
            bashrcpath=dic[cudaversion]['gputinkerbashrc']
        else:
            accept=False
            WriteToLogFile('node '+node+' with cudaversion '+str(cudaversion)+' has no bashrcpath') 
    else:
        bashrcpath=dic['cputinkerbashrc']


    return bashrcpath,accept


jobinfo={}
jobinfo['logname']={}
jobinfo['scratch']={}
jobinfo['scratchspace']={}
jobinfo['jobpath']={}
jobinfo['ram']={}
jobinfo=ReadJobInfoFromFile(jobinfo,jobinfofilepath)# input job info
jobtoprocess={}
if os.path.isfile(pidfile): # dont rerun daemon if instance is already running!
    head,tail=os.path.split(jobinfofilepath)
    tempfilepath=writepath+tail.replace('.txt','_TEMP.txt') #if daemon already running but still want to submit, submit to temporary file that will be read in by existing daemon process eventually
    WriteOutJobInfo(jobinfo,tempfilepath,jobtoprocess)
    sys.exit()
else:
    WritePIDFile() # identifies daemon has started
    AddJobInfoToDictionary(jobinfo,jobtoinfo,jobtoprocess)
    try:
        jobtologhandle={}
        cpunodetojoblist={} 
        gpunodetojoblist={}
        jobinfo=ReadJobInfoFromFile(jobinfo,jobtoinfo) # internal use text file jobtoinfo (the queue)
        ostocudaversiontobashrcpaths=ReadInBashrcs(bashrcfilename)
        cpujobs,gpujobs=PartitionJobs(jobinfo,cpuprogramlist,gpuprogramlist,jobtoprocess)
        jobtologhandle=CreateNewLogHandles(jobinfo['logname'],jobtologhandle)
        cpunodes,gpucards,nodetoosversion,gpunodetocudaversion=GrabCPUGPUNodes()
        cpunodetojoblist=DistributeJobsToNodes(cpunodes,cpujobs,jobinfo['scratchspace'],cpunodetojoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'])
        gpunodetojoblist=DistributeJobsToNodes(gpucards,gpujobs,jobinfo['scratchspace'],gpunodetojoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'])
        SubmitJobs(cpunodetojoblist,gpunodetojoblist,inputbashrcpath,sleeptime,jobtologhandle,jobinfo,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        
    finally:
        if os.path.isfile(pidfile): # delete pid file
            os.remove(pidfile)
    
