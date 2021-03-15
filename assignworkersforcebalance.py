import os
import sys
import subprocess
import shutil

# masterhost is NOVA, run program on NOVA
# Make sure TINKERPATH to binaries is in envpath file for each node.
# make sure analyze,minimize,dynamic and dynamic_omm have symbolic links in TINKERPATH directory for all nodes, since ForceBalance uses analyze and dynamic rather than dynamic_omm.x or analyze.x. For example (link -s dynamic_omm.x dynamic_omm, ln -s /home/liuchw/Softwares/tinkers/Tinker-latest/source-C8/analyze.x analyze,ln -s /home/liuchw/Softwares/tinkers/Tinker-latest/source-C8/minimize.x minimize,ln -s /home/liuchw/Softwares/tinkers/Tinker-latest/source-C8/dynamic.x dynamic)
# nohup python assignworkersforcebalance.py /home/bdw2292/.FB.bashrc /home/bdw2292/ExternalAPIRenLab/availablegpunodes.txt /home/bdw2292/FB_Test/example.in 9123 nova /home/bdw2292/.poltype_OS_6-7.bashrc &

fbbashrcpath=sys.argv[1]
availablenodeslist=sys.argv[2]
forcebalanceinputfilepath=sys.argv[3]
portnumber=int(sys.argv[4])
masterhost=sys.argv[5]
masterhostenvpath=sys.argv[6]
loghandle=open('logger.txt','a')

def ReadAvailableNodes(availablenodeslist):
    nodetoenvpath={}
    temp=open(availablenodeslist,'r')
    results=temp.readlines()
    temp.close()
    for line in results:
        linesplit=line.split()
        node=linesplit[0]
        bashrcpath=linesplit[1]
        nodetoenvpath[node]=bashrcpath 
    return nodetoenvpath

def AssignWorkers(fbbashrcpath,forcebalanceinputfilepath,nodetoenvpath,portnumber,loghandle,masterhost,masterhostenvpath):
    head,tail=os.path.split(forcebalanceinputfilepath)
    path=head
    AddPortnumberInputFile(forcebalanceinputfilepath,portnumber)
    CallForceBalance(fbbashrcpath,masterhost,masterhostenvpath,forcebalanceinputfilepath,loghandle,path,masterhost,portnumber,worker=False)

    for node,envpath in nodetoenvpath.items():
        CallForceBalance(fbbashrcpath,node,envpath,forcebalanceinputfilepath,loghandle,path,masterhost,portnumber,worker=True)

def AddPortnumberInputFile(f,portnumber):
    tempname='temp.in'
    temp=open(f,'r')
    results=temp.readlines()
    temp.close()
    temp=open(tempname,'w')
    enddelim='$end'
    count=0
    for line in results:
        if enddelim in line:
            if count>0:
                temp.write(enddelim+'\n')
                continue # assume $options at top
            count+=1
            temp.write('WQ_PORT '+str(portnumber)+'\n')
            temp.write(enddelim+'\n')
        else:
            temp.write(line)
    os.remove(f)
    os.rename(tempname,f)


def SpecifyGPUCard(cardvalue,job):
    string='CUDA_DEVICE_ORDER=PCI_BUS_ID '+';'+' export CUDA_VISIBLE_DEVICES='+str(cardvalue)
    job=string+';'+job
    return job



def CallForceBalance(fbbashrcpath,node,envpath,inputfilename,loghandle,path,masterhost,portnumber,worker=False):
    if masterhost[-1].isdigit() and '-' in masterhost:
        cardvalue=masterhost[-1]
        masterhost=masterhost[:-2]

    if worker==False:
        cmdstr='ForceBalance '+inputfilename
    else:
        cmdstr='work_queue_worker '+str(masterhost)+' '+str(portnumber)
    if node[-1].isdigit() and '-' in node:
        cardvalue=node[-1]
        node=node[:-2]
        cmdstr=SpecifyGPUCard(cardvalue,cmdstr)
    if worker==True:
        cmdstr = 'ssh %s "source %s;source %s;cd %s ;%s"' %(str(node),fbbashrcpath,envpath,path,cmdstr)
    else:
        cmdstr = "source %s;source %s;cd %s ;%s" %(fbbashrcpath,envpath,path,cmdstr)

    loghandle.write('Calling: '+cmdstr+'\n')
    process = subprocess.Popen(cmdstr, stdout=loghandle,stderr=loghandle,shell=True)

nodetoenvpath=ReadAvailableNodes(availablenodeslist)
AssignWorkers(fbbashrcpath,forcebalanceinputfilepath,nodetoenvpath,portnumber,loghandle,masterhost,masterhostenvpath)
