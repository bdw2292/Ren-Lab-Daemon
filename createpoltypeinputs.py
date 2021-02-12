import os
import sys

# python createpoltypeinputs.py /home/bdw2292/TestDaemonSubmissionScript/TestMolecules /home/bdw2292/TestDaemonSubmissionScript /home/bdw2292/currentstable/PoltypeModules/poltype.py 15GB 100GB 4
# nohup python /home/bdw2292/ExternalAPIRenLab/submit.py --jobinfofilepath=/home/bdw2292/TestDaemonSubmissionScript/jobinfo.txt &

structurefolderpath=sys.argv[1] # Folder with folders that have .sdf file in each folder
outputlogdirectory=sys.argv[2]
poltypepath=sys.argv[3]
ramalljobs=sys.argv[4]
diskalljobs=sys.argv[5]
numprocalljobs=sys.argv[6]
outputlogname='outputlog.txt'
inputdaemonfilename='jobinfo.txt'
outputlogpath=os.path.join(outputlogdirectory,outputlogname)
inputdaemonfilepath=os.path.join(outputlogdirectory,inputdaemonfilename)


def GrabStructuresAndJobPaths(structurefolderpath,poltypepath):
    os.chdir(structurefolderpath)
    files=os.listdir()
    jobpaths=[]
    structurelist=[]
    joblist=[]
    for f in files:
        path=os.path.join(structurefolderpath,f)
        os.chdir(path)
        subfiles=os.listdir()
        curdir=os.getcwd()
        for subf in subfiles:
            if '.sdf' in subf:
                structurelist.append(subf)
                jobpaths.append(curdir)
                joblist.append('cd '+curdir+' '+'&&'+' '+'python '+poltypepath)

    return jobpaths,structurelist,joblist
    

def GenerateDaemonInput(joblist,outputlogpath,scratchspacelist,ramlist,numproclist,jobpaths,inputdaemonfilepath,structurelist):
    head,tail=os.path.split(outputlogpath)
    os.chdir(head)
    temp=open(inputdaemonfilepath,'w')
    for i in range(len(joblist)):
        job=joblist[i]
        scratchspace=scratchspacelist[i]
        ram=ramlist[i]
        numproc=numproclist[i]
        jobpath=jobpaths[i]
        structure=structurelist[i]
        WritePoltypeInputFile(structure,jobpath,numproc,ram,scratchspace)
        string='--job='+job+' '+'--outputlogpath='+outputlogpath+' '+'--scratchspace='+str(scratchspace)+' '+'--ram='+str(ram)+' '+'--numproc='+str(numproc)+' '+'--jobpath='+jobpath+'\n'
        temp.write(string)
    temp.close()


def WritePoltypeInputFile(structure,jobpath,numproc,ram,scratchspace):
    curdir=os.getcwd()
    os.chdir(jobpath)
    newtemp=open('poltype.ini','w')
    newtemp.write('structure='+structure+'\n')
    newtemp.write('maxmem='+ram+'\n')
    newtemp.write('maxdisk='+scratchspace+'\n')
    newtemp.write('numproc='+numproc+'\n')
    newtemp.close()
    os.chdir(curdir)


def ResourceInputs(jobpaths,ramalljobs,diskalljobs,numprocalljobs):
    scratchspacelist=[]
    ramlist=[]
    numproclist=[]
    for i in range(len(jobpaths)):
        scratchspacelist.append(diskalljobs)
        ramlist.append(ramalljobs)
        numproclist.append(numprocalljobs)

    return scratchspacelist,ramlist,numproclist


jobpaths,structurelist,joblist=GrabStructuresAndJobPaths(structurefolderpath,poltypepath)
scratchspacelist,ramlist,numproclist=ResourceInputs(jobpaths,ramalljobs,diskalljobs,numprocalljobs)
GenerateDaemonInput(joblist,outputlogpath,scratchspacelist,ramlist,numproclist,jobpaths,inputdaemonfilepath,structurelist)

