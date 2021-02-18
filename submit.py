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
global preserveresourceratio
global listavailnodes
global availablecpunodesname
global availablegpunodesname
global loggersleeptime
global currenttime
global redocrashedjobs
global masterfinishedloghandle


curdir = os.path.dirname(os.path.realpath(__file__))+r'/'
bashrcfilename=curdir+'tinkerbashrcpaths.txt'
nodelistfilepath=curdir+'nodes.txt'
inputbashrcpath=None
jobinfofilepath=None
pidfile=curdir+'daemon.pid' # just in case want to kill daemon
loggerfile=curdir+'logger.txt'
errorloggerfile=curdir+'errorlogger.txt'
finishedloggerfile=curdir+'finishedlogger.txt'

jobtoinfo=curdir+'jobtoinfo.txt'
writepath=curdir

masterloghandle=open(loggerfile,'w',buffering=1)
mastererrorloghandle=open(errorloggerfile,'a',buffering=1)
masterfinishedloghandle=open(finishedloggerfile,'a',buffering=1)

sleeptime=60
cpuprogramexceptionlist=['psi4','g09','g16',"cp2k.ssmp","mpirun_qchem","dynamic.x"] # dont run on cpu node if detect these programs
restrictedprogramtonumber={'bar.x 1':10,'bar_omm.x 1':10} # restrictions on program use for network slow downs
currentrestrictedprogramtoprocesslist={'bar.x 1':[],'bar_omm.x 1':[]}
cpuprogramlist=['psi4','g09','g16',"cp2k.ssmp","mpirun_qchem","dynamic.x",'minimize.x','minimize','poltype.py'] # run on cpu node env if see these
gpuprogramlist=['dynamic_omm.x','bar_omm.x','dynamic.gpu' , 'dynamic.mixed' , 'analyze.gpu' , 'analyze.mixed' , 'bar.gpu', 'bar.mixed'] # run on gpu node env if see these
cpunodesonly=False
gpunodesonly=False
nodetimeout=10 # nodetime out if checking node features with command stalls
verbosemode=False
preserveresourceratio=.2
listavailnodes=False
availablecpunodesname='availablecpunodes.txt'
availablegpunodesname='availablegpunodes.txt'
loggersleeptime=60*5
currenttime=time.time()
redocrashedjobs=False


opts, xargs = getopt.getopt(sys.argv[1:],'',["redocrashedjobs","listavailnodes","bashrcpath=","jobinfofilepath=","cpunodesonly","gpunodesonly","verbosemode"])
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
    elif o in ("--listavailnodes"):
        listavailnodes=True
    elif o in ("--redocrashedjobs"):
        redocrashedjobs=True



def AlreadyActiveNodes(nodelist,programexceptionlist=None,gpunodes=False):
    activenodelist=[]
    if verbosemode==True:
        disablebool=False
    else:
        disablebool=True

    for nodeidx in tqdm(range(len(nodelist)),desc='Checking already active nodes',disable=disablebool):
        node=nodelist[nodeidx]
        keepnode=True
        nonactivecards=[]
        if gpunodes==True:
            activecards=CheckWhichGPUCardsActive(node)
            for card in activecards:
                activenodelist.append(node+'-'+str(card))

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
                    if verbosemode==True:
                        WriteToLogFile(output1)
                if type(output2)==str:
                    if verbosemode==True:
                        WriteToLogFile(output2)

        if keepnode==True:
            pass 
        else:
            activenodelist.append(node)
            if verbosemode==True:
                WriteToLogFile('cannot use cpunode '+node+' because of program exception')

    return activenodelist

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
    activecards=[]
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
                    pass
                else:
                    card=node+'-'+str(count)
                    activecards.append(count)
                    if verbosemode==True:
                        WriteToLogFile('GPU card '+card+' is currently active and cannot use')
            else:
                card=node+'-'+str(count)
                if verbosemode==True:
                    WriteToLogFile('GPU card '+card+' has problem in nvidia-smi and cannot use')


    return activecards

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
                if verbosemode==True:
                    WriteToLogFile('Removing from node list '+newline)    

        temp.close()
    return nodelist,cpunodesonlylist,gpunodesonlylist


def CheckScratchSpaceAllNodes(nodes):
    nodetoscratchspace={}
    for node in nodes:
        scratchavail=CheckScratchSpace(node)
        if scratchavail!=False:
            nodetoscratchspace[node]=scratchavail

    return nodetoscratchspace

def CheckBashrcPathsAllNodes(nodes,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths):
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
    amounttopreserve=preserveresourceratio*ram
    ram=ram-amounttopreserve

    return ram

def CheckCPUsAllNodes(nodes):
    nodetonumproc={}
    for node in nodes:
        numproc=CheckCPUs(node)
        if numproc!=False:
            nodetonumproc[node]=numproc
    return nodetonumproc


def CheckCPUs(node):
    proc=False
    totalproc=CheckTotalCPU(node)
    currentproc=CheckCurrentCPUUsage(node)
    if type(totalproc)==int and type(currentproc)==int:
        proc=totalproc-currentproc
        amounttopreserve=int(preserveresourceratio*proc)
        proc=proc-amounttopreserve
    return proc

def CheckTotalCPU(node):
    totalproc=False
    cmdstr='nproc'
    output=CheckOutputFromExternalNode(node,cmdstr)
    if output!=False:
        lines=output.split('\n')
        firstline=lines[0]
        firstlinesplit=firstline.split()
        totalproc=int(firstlinesplit[0])
    return totalproc

def CheckCurrentCPUUsage(node):
    currentproc=False
    filepath=os.path.join(os.getcwd(),'topoutput.txt')
    cmdstr='top -b -n 1 > '+filepath   
    job='ssh %s "%s"'%(node,cmdstr)
    p = subprocess.Popen(job, stdout=subprocess.PIPE,shell=True)
    nodedead,output=CheckForDeadNode(p,node)
    if nodedead==False:
        if os.path.isfile(filepath):
            temp=open(filepath,'r')
            results=temp.readlines()
            temp.close()
            procsum=0
            for line in results:
                linesplit=line.split()
                if len(linesplit)==12:
                   proc=linesplit[8]
                   if proc.isnumeric():
                       proc=float(proc)/100
                       procsum+=proc
            currentproc=int(procsum)
            os.remove(filepath)
    return currentproc


def CheckRAMAllNodes(nodes):
    nodetoram={}
    for node in nodes:
        ramavail=CheckRAM(node)
        if ramavail!=False:
            nodetoram[node]=ramavail
    return nodetoram

    
def CheckIfEnoughRAM(ramneeded,ramavail):
    enough=False
    if ramavail>float(ramneeded):
        enough=True
    return enough


def AssignPossibleNodesTojobs(nodes,nodetoram,nodetoscratchspace,nodetonumproc,jobs,jobtoscratchspace,jobtoram,jobtonumproc,cpujobs):
    jobtopossiblenodes={}
    for job in jobs:
        possiblenodes=[]
        if cpujobs==True:
            ramneeded=jobtoram[job]
            scratchneeded=jobtoscratchspace[job]
            numprocneeded=jobtonumproc[job]
            for node,ram in nodetoram.items():
                scratchspace=nodetoscratchspace[node]
                numproc=nodetonumproc[node]
             
                goodnode=CheckIfNodeHasEnoughResources(scratchneeded,scratchspace,ramneeded,ram,numprocneeded,numproc)
                if goodnode==True:
                    possiblenodes.append(node)
        else:
            for node in nodes:
                possiblenodes.append(node) 
        jobtopossiblenodes[job]=possiblenodes
    return jobtopossiblenodes

def CheckIfNodeHasEnoughResources(scratchneeded,scratchspace,ramneeded,ram,numprocneeded,numproc):
    boolarray=[]
    if ramneeded!=None:
        ramneeded=ConvertMemoryToGBValue(ramneeded)
        enoughram=CheckIfEnoughRAM(ramneeded,ram)
        boolarray.append(enoughram)
    if scratchneeded!=None:
        scratchneeded=ConvertMemoryToGBValue(scratchneeded)
        enoughscratch=CheckIfEnoughScratch(scratchspace,scratchneeded)
        boolarray.append(enoughscratch)
    if numprocneeded!=None:
        numprocneeded=int(numprocneeded)
        enoughnumproc=CheckIfEnoughCPU(numproc,numprocneeded)
        boolarray.append(enoughnumproc)
    goodnode=True
    for boolarg in boolarray:
        if boolarg==False:
            goodnode=False
    return goodnode

def CheckIfEnoughCPU(numproc,numprocneeded):
    enoughnumproc=False
    if numproc>numprocneeded:
        enoughnumproc=True
    return enoughnumproc


def AssignNodesToJobs(jobtopossiblenodes,nodetoram,nodetoscratchspace,nodetonumproc,jobtoram,jobtoscratchspace,jobtonumproc,activenodes,cpujobs):
    jobtonode={}
    currentnodeassignedlist=[]
    for job,possiblenodes in jobtopossiblenodes.items():
        for node in possiblenodes:
            if cpujobs==True:
                ram=nodetoram[node]
                scratchspace=nodetoscratchspace[node]
                numproc=nodetonumproc[node]
                ramneeded=jobtoram[job]
                scratchneeded=jobtoscratchspace[job]
                numprocneeded=jobtonumproc[job]
                if ramneeded==None and scratchneeded==None and numprocneeded==None:
                    if node not in currentnodeassignedlist and node not in activenodes:
                        currentnodeassignedlist.append(node)
                        jobtonode[job]=node
                        break

                else:
                    scratchneeded=ConvertMemoryToGBValue(scratchneeded)
                    ramneeded=ConvertMemoryToGBValue(ramneeded)
                    numprocneeded=int(numprocneeded)
                    nodetoscratchspace[node]=scratchspace-scratchneeded
                    nodetoram[node]=ram-ramneeded
                    nodetonumproc[node]=numproc-numprocneeded
                    jobtonode[job]=node

                    break
            else:
                if node not in currentnodeassignedlist and node not in activenodes:
                    currentnodeassignedlist.append(node)
                    jobtonode[job]=node
                    break
    return jobtonode


def ReverseDictionary(keytovalue):
    valuetokeylist={}
    for key,value in keytovalue.items():
        if value not in valuetokeylist.keys():
            valuetokeylist[value]=[]
        if key not in valuetokeylist[value]:
            valuetokeylist[value].append(key)
    return valuetokeylist


def FilterDictionariesWithoutSameKeys(nodetoram,nodetoscratchspace,nodetonumproc):
    minlist=[list(nodetoram.keys()),list(nodetoscratchspace.keys()),list(nodetonumproc.keys())]
    minlenlist=[len(i) for i in minlist]
    minvalue=min(minlenlist)
    minidx=minlenlist.index(minvalue)
    minkeylist=minlist[minidx] 
    delkeys=[]
    for key in nodetoram.keys():
        if key not in minkeylist:
            delkeys.append(key)
    for key in delkeys:
        del nodetoram[key]    
    delkeys=[]
    for key in nodetoscratchspace.keys():
        if key not in minkeylist:
            delkeys.append(key)
    for key in delkeys:
        del nodetoscratchspace[key]    
 
    delkeys=[]
    for key in nodetonumproc.keys():
        if key not in minkeylist:
            delkeys.append(key)
    for key in delkeys:
        del nodetonumproc[key]    

    return nodetoram,nodetoscratchspace,nodetonumproc

def WriteAvailableNodesOut(nodes,cpujobs,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,activenodes):
    if cpujobs==True:
        avail=availablecpunodesname
    else:
        avail=availablegpunodesname
    handle=open(avail,'w')
    for node in nodes:
        bashrcpath,accept=DetermineBashrcPath(nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,node)
        if node not in activenodes:
            handle.write(node+' '+bashrcpath+'\n')
    handle.close()

def DistributeJobsToNodes(nodes,jobs,jobtoscratchspace,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobtoram,jobtonumproc,activenodes,cpujobs=False):
    nodetojoblist={}
    if len(nodes)!=0:
        newnodes=CheckBashrcPathsAllNodes(nodes,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        if listavailnodes==True:
            WriteAvailableNodesOut(newnodes,cpujobs,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,activenodes)
            if os.path.isfile(availablecpunodesname) and os.path.isfile(availablegpunodesname):
                sys.exit()

        if cpujobs==True:
            nodetoram=CheckRAMAllNodes(newnodes)
            newnodes=list(nodetoram.keys())
            nodetoscratchspace=CheckScratchSpaceAllNodes(newnodes)
            newnodes=list(nodetoscratchspace.keys())
            nodetonumproc=CheckCPUsAllNodes(newnodes)
            nodetoram,nodetoscratchspace,nodetonumproc=FilterDictionariesWithoutSameKeys(nodetoram,nodetoscratchspace,nodetonumproc)
        else:
            nodetoram={}
            nodetoscratchspace={}
            nodetonumproc={}
        jobtopossiblenodes=AssignPossibleNodesTojobs(newnodes,nodetoram,nodetoscratchspace,nodetonumproc,jobs,jobtoscratchspace,jobtoram,jobtonumproc,cpujobs)
        jobtonode=AssignNodesToJobs(jobtopossiblenodes,nodetoram,nodetoscratchspace,nodetonumproc,jobtoram,jobtoscratchspace,jobtonumproc,activenodes,cpujobs)
        nodetojoblist=ReverseDictionary(jobtonode)
        for node,joblist in nodetojoblist.items():
            for job in joblist:
                if verbosemode==True:
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
            if verbosemode==True:
                WriteToLogFile(' node '+node+' has no scratch')
        if scratchavail==False:
            cmdstr="du -h /scratch | sort -n -r | head -n 15"
            output=CheckOutputFromExternalNode(node,cmdstr)
            if verbosemode==True:
                WriteToLogFile(output)
        else:
            scratchavail=ConvertMemoryToGBValue(scratchavail)
            amounttopreserve=preserveresourceratio*scratchavail
            scratchavail=scratchavail-amounttopreserve

        


    return scratchavail

def ConvertMemoryToGBValue(scratch):
    availspace,availunit=SplitScratch(scratch)
    if availunit=='M' or availunit=='MB':
        availspace=float(availspace)*.001
    elif availunit=='T' or availunit=='TB':
        availspace=float(availspace)*1000
    elif availunit=='G' or availunit=='GB':
        availspace=float(availspace)
    return availspace
     

def CheckIfEnoughScratch(scratchspace,scratchneeded):
    enoughscratchspace=False
    if scratchspace>scratchneeded:
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
        jobinfo=RemoveJobInfoFromQueue(jobinfo,jobtoprocess)
    return jobtoprocess,jobinfo

def PollProcess(jobtoprocess,job,finishedjoblist,loghandle,polledjobs,currenttime,mastererrorloghandle):
    process=jobtoprocess[job]
    poll=process.poll()
    polledjobs.append(job)
    jobstodelete=[]
    now=time.time()
    diff=now-currenttime
    normaltermjobs=[]
    if poll!=None:
        out, err = process.communicate()
        finishedjoblist.append(job)
        WriteToLogFile(job+' '+'has terminated',loghandle)
        if process.returncode != 0:
            WriteToLogFile('Error detected for job '+job,loghandle=mastererrorloghandle)
        else:
            WriteToLogFile('Normal termination for job '+job,loghandle=masterfinishedloghandle)
            normaltermjobs.append(job)
        jobstodelete.append(job)
        for program in restrictedprogramtonumber.keys():
            if program in job:
                plist=currentrestrictedprogramtoprocesslist[program]
                if process in plist:
                    currentrestrictedprogramtoprocesslist[program].remove(process)
        
    else:
        for program in restrictedprogramtonumber.keys():
            if program in job:
                plist=currentrestrictedprogramtoprocesslist[program]
                if process not in plist:
                    currentrestrictedprogramtoprocesslist[program].append(process) 
        if diff>=loggersleeptime:
            WriteToLogFile(job+' '+'has not terminated ',loghandle)
            currenttime=time.time()

    return finishedjoblist,polledjobs,jobstodelete,normaltermjobs,currenttime,mastererrorloghandle


def SubmitJobs(cpunodetojoblist,gpunodetojoblist,inputbashrcpath,sleeptime,jobtologhandle,jobinfo,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,currenttime,mastererrorloghandle,masterfinishedloghandle):
    jobnumber=len(jobtologhandle.keys())
    jobtoprocess={}
    finishedjoblist=[]
    while len(finishedjoblist)!=jobnumber:
        jobtoprocess,jobinfo=SubmitJobsLoop(cpunodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        jobtoprocess,jobinfo=SubmitJobsLoop(gpunodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths)
        finishedjoblist,jobtoprocess,currenttime,mastererrorloghandle=CheckJobTermination(jobtoprocess,finishedjoblist,jobtologhandle,currenttime,mastererrorloghandle)
        time.sleep(sleeptime)
        WriteToLogFile('*************************************')
        cpunodes,gpucards,nodetoosversion,gpunodetocudaversion,cpunodesactive,gpucardsactive=GrabCPUGPUNodes()
        jobinfo,mastererrorloghandle,masterfinishedloghandle=ReadTempJobInfoFiles(jobinfo,mastererrorloghandle,masterfinishedloghandle)
        mastererrorloghandle,masterfinishedloghandle=AddJobInfoToDictionary(jobinfo,jobtoinfo,jobtoprocess,mastererrorloghandle,masterfinishedloghandle)
        jobinfo,mastererrorloghandle,masterfinishedloghandle=ReadJobInfoFromFile(jobinfo,jobtoinfo,mastererrorloghandle,masterfinishedloghandle)
        cpujobs,gpujobs=PartitionJobs(jobinfo,cpuprogramlist,gpuprogramlist,jobtoprocess)
        jobtologhandle=CreateNewLogHandles(jobinfo['logname'],jobtologhandle)
        cpunodetojoblist=DistributeJobsToNodes(cpunodes,cpujobs,jobinfo['scratchspace'],nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'],jobinfo['numproc'],cpunodesactive,True)
        gpunodetojoblist=DistributeJobsToNodes(gpucards,gpujobs,jobinfo['scratchspace'],nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'],jobinfo['numproc'],gpucardsactive,False)
        jobnumber=len(jobtologhandle.keys())
    WriteToLogFile('All jobs have finished ')

def SubmitJobsLoop(nodetojoblist,jobtologhandle,jobinfo,jobtoprocess,finishedjoblist,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths):
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
            if job not in jobtoprocess.keys() and job not in finishedjoblist:
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
                    jobtoprocess,jobinfo=SubmitJob(node,path,bashrcpath,job,loghandle,jobtoprocess,jobinfo['scratch'],jobinfo)
    return jobtoprocess,jobinfo


def CheckJobTermination(jobtoprocess,finishedjoblist,jobtologhandle,currenttime,mastererrorloghandle):
    polledjobs=[]
    dellist=[]
    totalnormaltermjobs=[]
    for job in jobtoprocess.keys():
        if job not in polledjobs and job not in finishedjoblist:
            loghandle=jobtologhandle[job]
            finishedjoblist,polledjobs,jobtodelete,normaltermjobs,currenttime,mastererrorloghandle=PollProcess(jobtoprocess,job,finishedjoblist,loghandle,polledjobs,currenttime,mastererrorloghandle)
            dellist.extend(jobtodelete)
            totalnormaltermjobs.extend(normaltermjobs)
    for job in dellist:
        del jobtoprocess[job]

    mastererrorloghandle=RemoveNormalTermJobsFromErrorLog(totalnormaltermjobs,mastererrorloghandle) # if redocrashedjobs and normal term remove from log file
    return finishedjoblist,jobtoprocess,currenttime,mastererrorloghandle

def GPUCardToNode(gpucards,cpunodesonlylist):
    gpucardtonode={}
    for gpucard in gpucards:
        gpunode=gpucard[:-2]
        if gpunode not in cpunodesonlylist:
            gpucardtonode[gpucard]=gpunode
        else:
            if verbosemode==True:
                WriteToLogFile('removing node '+gpunode+' from gpunode list since its marked as CPUONLY')

    return gpucardtonode

def SpecifyGPUCard(cardvalue,job):
    string='CUDA_DEVICE_ORDER=PCI_BUS_ID '+';'+' export CUDA_VISIBLE_DEVICES='+str(cardvalue)
    job=string+';'+job
    return job


def RemoveJobInfoFromQueue(jobinfo,jobtoprocess):
    newjobinfo=RemoveAlreadySubmittedJobs(jobtoprocess,jobinfo) # just removing submissions from queue
    WriteOutJobInfo(newjobinfo,jobtoinfo,jobtoprocess)
    return newjobinfo 

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
    jobtonumproc=jobinfo['numproc']
    counter=0
    array=['--scratchdir=','--scratchspace=','--jobpath=','--ram=','--numproc=']
    for job,log in jobtologname.items():
        if job in jobtoprocess.keys():
            continue
        counter+=1
        scratch=jobtoscratch[job]
        scratchspace=jobtoscratchspace[job]
        jobpath=jobtojobpath[job]
        ram=jobtoram[job]
        numproc=jobtonumproc[job]
        curarray=[scratch,scratchspace,jobpath,ram,numproc]
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


def AddJobInfoToDictionary(jobinfo,filepath,jobtoprocess,mastererrorloghandle,masterfinishedloghandle):
    jobinfoprev,mastererrorloghandle,masterfinishedloghandle=ReadJobInfoFromFile(jobinfo,jobtoinfo,mastererrorloghandle,masterfinishedloghandle)
    for key in jobinfoprev.keys():
        prevd=jobinfoprev[key]
        jobinfo[key].update(prevd)
    WriteOutJobInfo(jobinfo,filepath,jobtoprocess)
    return mastererrorloghandle,masterfinishedloghandle


def ParseJobInfo(line):
    linesplit=line.split('--')[1:]
    linesplit=[e.rstrip() for e in linesplit]
    job=None
    logname=None
    scratch=None
    scratchspace=None
    jobpath=None
    ram=None
    numproc=None
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
            ram=line.replace('ram=','')
        if "numproc=" in line:
            numproc=line.replace('numproc=','')

    return job,logname,scratch,scratchspace,jobpath,ram,numproc


def ReadTempJobInfoFiles(jobinfo,mastererrorloghandle,masterfinishedloghandle):
    curdir=os.getcwd()
    os.chdir(writepath)
    files=os.listdir()
    dellist=[]
    for f in files:
        if '_TEMP' in f:
            jobinfo,mastererrorloghandle,masterfinishedloghandle=ReadJobInfoFromFile(jobinfo,f,mastererrorloghandle,masterfinishedloghandle)
            dellist.append(f)
    for f in dellist:
        if os.path.isfile(f):
            os.remove(f)
    return jobinfo,mastererrorloghandle,masterfinishedloghandle

   
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


def ReadCrashedJobs(mastererrorloghandle):
    crashedjobs=[]
    if redocrashedjobs==True:
        mastererrorloghandle.close()
        temp=open(errorloggerfile,'r')
        results=temp.readlines()
        temp.close()
        delim='Error detected for job'
        for line in results:
            linesplit=line.split(delim)
            job=linesplit[-1].lstrip().rstrip()
            crashedjobs.append(job)
         
        mastererrorloghandle=open(errorloggerfile,'a',buffering=1)

    return crashedjobs,mastererrorloghandle


def ReadFinishedJobs(masterfinishedloghandle):
    finishedjobs=[]
    masterfinishedloghandle.close()
    temp=open(finishedloggerfile,'r')
    results=temp.readlines()
    temp.close()
    delim='Normal termination for job'
    for line in results:
        linesplit=line.split(delim)
        job=linesplit[-1].lstrip().rstrip()
        finishedjobs.append(job)
     
    masterfinishedloghandle=open(finishedloggerfile,'a',buffering=1)

    return finishedjobs,masterfinishedloghandle  




def RemoveNormalTermJobsFromErrorLog(totalnormaltermjobs,mastererrorloghandle):
    mastererrorloghandle.close()
    temp=open(errorloggerfile,'r')
    results=temp.readlines()
    temp.close()
    newlines=[]
    for line in results:
        keep=True
        for job in totalnormaltermjobs:
            if job in line:
                keep=False
        
        if keep==True:
            newlines.append(line)
    os.remove(errorloggerfile)
    mastererrorloghandle=open(errorloggerfile,'a',buffering=1)
    for line in newlines:
        mastererrorloghandle.write(line)
    return mastererrorloghandle 


def ReadJobInfoFromFile(jobinfo,filename,mastererrorloghandle,masterfinishedloghandle):
    if filename==None:
        return jobinfo,mastererrorloghandle,masterfinishedloghandle

    crashedjobs,mastererrorloghandle=ReadCrashedJobs(mastererrorloghandle) 
    finishedjobs,masterfinishedloghandle=ReadFinishedJobs(masterfinishedloghandle)
    WriteToLogFile('crashed jobs '+str(crashedjobs))
    WriteToLogFile('finishedjobs '+str(finishedjobs))
    if os.path.isfile(filename):
        temp=open(filename,'r')
        results=temp.readlines()
        temp.close()
        addjob=False
        for line in results:
            job,log,scratch,scratchspace,jobpath,ram,numproc=ParseJobInfo(line)
            if job==None or log==None:
                continue
            if len(crashedjobs)>0:
                if job in crashedjobs:
                    addjob=True
            else:
                addjob=True
            if job in finishedjobs:
                addjob=False
            if addjob==True:
                jobinfo['logname'][job]=log
                jobinfo['scratch'][job]=scratch
                jobinfo['scratchspace'][job]=scratchspace
                jobinfo['jobpath'][job]=jobpath
                jobinfo['ram'][job]=ram
                jobinfo['numproc'][job]=numproc


    return jobinfo,mastererrorloghandle,masterfinishedloghandle



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
    WriteToLogFile("Checking active CPU nodes")
    newcpunodes=[]
    for cpunode in cpunodes:
        if cpunode not in gpunodesonlylist:
            newcpunodes.append(cpunode)
        else:
            if verbosemode==True:
                WriteToLogFile('removing node '+cpunode+' from cpunode list since its marked as GPUONLY')
    if gpunodesonly==False:
        cpunodesactive=AlreadyActiveNodes(newcpunodes,cpuprogramexceptionlist)
    else:
        newcpunodes=[]
        cpunodesactive=[]
    
    gpucardtonode=GPUCardToNode(gpucards,cpunodesonlylist)
    gpucards=list(gpucardtonode.keys())
    gpunodes=list(set(gpucardtonode.values()))
    WriteToLogFile('*************************************')
    WriteToLogFile("Checking active GPU cards")
    if cpunodesonly==False:
        gpucardsactive=AlreadyActiveNodes(gpunodes,programexceptionlist=None,gpunodes=True)
    else:
        gpucards=[]
        gpucardsactive=[]
    for gpucard in gpucards:
        if verbosemode==True:
            WriteToLogFile('GPU card '+gpucard+' is available for submission')
    for cpunode in newcpunodes:
        if verbosemode==True:
            WriteToLogFile('CPU node '+cpunode+' is avaible for submission')
    return newcpunodes,gpucards,nodetoosversion,gpunodetocudaversion,cpunodesactive,gpucardsactive

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
    if osversion in nodetoosversion.values():
        dic=ostocudaversiontobashrcpaths[osversion]
        if usinggpunode==True:
            cudaversion=gpunodetocudaversion[gpunode]
            if cudaversion in dic.keys():
                bashrcpath=dic[cudaversion]['gputinkerbashrc']
                if not os.path.isfile(bashrcpath):
                    accept=False
                    if verbosemode==True:
                        WriteToLogFile('bashrcpath '+bashrcpath+' does not exist')

            else:
                accept=False
                if verbosemode==True:
                    WriteToLogFile('node '+node+' with cudaversion '+str(cudaversion)+' has no bashrcpath') 
        else:
            bashrcpath=dic['cputinkerbashrc']
            if not os.path.isfile(bashrcpath):
                accept=False
                if verbosemode==True:
                    WriteToLogFile('bashrcpath '+bashrcpath+' does not exist')

    else:
        accept=False
        if verbosemode==True:
            WriteToLogFile('node '+node+' with OS version '+str(osversion)+' has no bashrcpath') 



    return bashrcpath,accept

jobinfo={}
jobinfo['logname']={}
jobinfo['scratch']={}
jobinfo['scratchspace']={}
jobinfo['jobpath']={}
jobinfo['ram']={}
jobinfo['numproc']={}
jobinfo,mastererrorloghandle,masterfinishedloghandle=ReadJobInfoFromFile(jobinfo,jobinfofilepath,mastererrorloghandle,masterfinishedloghandle)# input job info
jobtoprocess={}
if os.path.isfile(pidfile) and listavailnodes==False: # dont rerun daemon if instance is already running!
    head,tail=os.path.split(jobinfofilepath)
    tempfilepath=writepath+tail.replace('.txt','_TEMP.txt') #if daemon already running but still want to submit, submit to temporary file that will be read in by existing daemon process eventually
    WriteOutJobInfo(jobinfo,tempfilepath,jobtoprocess)
    sys.exit()
else:
    WritePIDFile() # identifies daemon has started
    mastererrorloghandle,masterfinishedloghandle=AddJobInfoToDictionary(jobinfo,jobtoinfo,jobtoprocess,mastererrorloghandle,masterfinishedloghandle)
    try:
        jobtologhandle={}
        cpunodetojoblist={} 
        gpunodetojoblist={}
        jobinfo,mastererrorloghandle,masterfinishedloghandle=ReadJobInfoFromFile(jobinfo,jobtoinfo,mastererrorloghandle,masterfinishedloghandle) # internal use text file jobtoinfo (the queue)
        ostocudaversiontobashrcpaths=ReadInBashrcs(bashrcfilename)
        cpujobs,gpujobs=PartitionJobs(jobinfo,cpuprogramlist,gpuprogramlist,jobtoprocess)
        jobtologhandle=CreateNewLogHandles(jobinfo['logname'],jobtologhandle)
        cpunodes,gpucards,nodetoosversion,gpunodetocudaversion,cpunodesactive,gpucardsactive=GrabCPUGPUNodes()
        cpunodetojoblist=DistributeJobsToNodes(cpunodes,cpujobs,jobinfo['scratchspace'],nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'],jobinfo['numproc'],cpunodesactive,True)
        gpunodetojoblist=DistributeJobsToNodes(gpucards,gpujobs,jobinfo['scratchspace'],nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,jobinfo['ram'],jobinfo['numproc'],gpucardsactive,False)
        SubmitJobs(cpunodetojoblist,gpunodetojoblist,inputbashrcpath,sleeptime,jobtologhandle,jobinfo,nodetoosversion,gpunodetocudaversion,ostocudaversiontobashrcpaths,currenttime,mastererrorloghandle,masterfinishedloghandle)
        
    finally:
        if os.path.isfile(pidfile): # delete pid file
            os.remove(pidfile)
    
