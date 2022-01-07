#!/usr/bin/python3
import argparse
import subprocess
import sys
import os
from cache import Cache
from concurrent.futures import ThreadPoolExecutor
from parser import Parser
from multiprocessing import cpu_count
import xml.etree.ElementTree as ET
from multiprocessing import Lock

def blame_command(path_to_file):
    date_format = "'%Y-%m-%d %H:%M:%S %z (%a, %d %b %Y)'"
    return 'git blame -c -w -l --date=format:%s --no-progress %s | sed \'s/\t/ /g\' | sed \'s/(//\'' \
        ' | sed \'s/).*)/) /\''\
       % (date_format, path_to_file)

class Blamer(object):
    def __init__(self, cache, jobs, project_root):
        self.__cache = cache
        self.__jobs = jobs
        self.__project_root = project_root
        self.__revisions_logged = set(cache.get_revisions())
        self.__hcmutex = Lock()
        print("Python version")
        print (sys.version)



    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        print("Blaming finished.")

        # pass

    def process_data(self, data):
        print("Processing files", flush=True)
        self.__run_threads(data)

    def __run_threads(self, data):
        with ThreadPoolExecutor(self.__jobs) as ex:
            ex.map(self.blame_file, data)

            print("split_revision")
        return subprocess.check_output(cmd, cwd=self.__project_root, shell=True).decode('utf-8', errors="ignore")

    def __get_git_logs(self, revision):
        revisions_log = "[{0}]".format(self.__get_git_logs_list_format(revision))
        if not revisions_log:
            return []
        try:
            return eval(revisions_log)
        except Exception as e:
            print(e)
            print(revisions_log)

    def __get_logs(self, revisions):
        return self.__get_git_logs(revisions)
       

    def __extract_from_xml(self, log_output):
        root = ET.fromstring(log_output)
        ret = []
        for logentry in root:
            rev = logentry.attrib["revision"]
            author = logentry[0].text
            msg = logentry[2].text
            ret.append([rev, author, msg])
        return ret

    def __get_git_blame(self, file):
        try:
            print(blame_command(file))
            process=subprocess.Popen(blame_command(file),cwd=self.__project_root,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
            #process=subprocess.Popen("git blame README.md" , cwd="/var/fpwork/chahe/gnb/ci-if",shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
            stdout, stderr = process.communicate()
            #git_info= subprocess.check_output(blame_command(file), cwd=self.__project_root, shell=True)
            print("git_info", stdout)
            return stdout
        except:
            return []        
        finally:
            process.terminate() # send sigterm, or ...
            process.kill()      # send sigkill

    def __blame_file(self, filename):
        #self.__hcmutex.acquire()
        blame_output= None
        blame_output = self.__get_git_blame(filename)
        try:
            if blame_output:
                print(filename)
                print(blame_output)
                blame_output = blame_output.decode('utf-8', errors="ignore").split('\n')
                print("blame_output %s" %filename)
                blame_output = [self.__parse_blame_line(i) for i in blame_output]#here is wrong
                self.__cache.add_blame(filename, blame_output)
                #HCX
                print("Blamed %s." % filename)
        except:
            pass
        #finally:
            #self.__hcmutex.release()


           
        return blame_output

    def __get_blame(self, filename):
        ret = self.__cache.get_blame(filename)
        return ret if ret is not None else self.__blame_file(filename)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--xml', type=str, required=True, help="XML file with coverage info")
    args.add_argument('--project_root', type=str, required=False, default='',
                      help="Path to the root of the project. Default value is current directory")
    args.add_argument('-j', '--jobs', type=int, required=False, default=8,
                      help="Number of threads to spawn. Default value is the number of threads in the cpu")
    args.add_argument('--cache_file', type=str, required=False, default='default.blame_cache',
                      help="File to use as cache. Default value is default.blame_cache")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.xml):
        exit(print(args.xml+" XML file does not exist."))
    else:
        print("Xml file exists. Parsing started.")

    if not os.path.exists(args.cache_file):
        print(args.cache_file + " cache file does not exist.")
    else:
        print("Cache file exists.")

    if not os.path.exists(args.project_root):
        exit(print("Project root does not exist."))
    else:
        print("Project root exists. Blaming started.")

    p = Parser(args.xml)
    dat = p.parse_xml_file()
    keys = list(dat.keys())

    try:
        print("the number of cpu is")
        print(args.jobs)
        print("We enable the multithreading feature")
        with Cache(args.cache_file, args.project_root) as cache:
            with Blamer(cache, args.jobs, args.project_root) as b:

                b.process_data(keys)
   
    except:
        os.remove(args.cache_file)    
        with Cache(args.cache_file, args.project_root) as cache:
                with Blamer(cache, args.jobs, args.project_root) as b:

                    b.process_data(keys)
