#!/usr/bin/env python3

import os
import time
import socket
import queue
import threading
from threading import Thread

import logging
import sys
import argparse

import subprocess
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
from string import Formatter
import sqlite3
import socket

mq = queue.Queue()
slot = dict()
active_ps = dict()
active = mp.Value("i", 0)
dbfile = "pardb.sqlite"


def log(*args, sep=" "):
    logging.debug(sep.join(map(str, args)))


def hello(counter: mp.Value):
    workerid = threading.get_native_id()
    with counter.get_lock():
        slot[workerid] = counter.value
        counter.value += 1
    affinity = None
    logging.debug(f"Worker: pid={os.getpid()} ID={counter.value}, TID={workerid}")
    return 0


def execute(verbose=False, dryrun=False):
    ## check in
    hostname = socket.gethostname()
    workerid = threading.get_native_id()

    while True:
        nomorejob = False
        with sqlite3.connect(dbfile) as con:
            while True:
                try:
                    con.execute("BEGIN EXCLUSIVE")
                    cur = con.cursor()
                    cur.execute(
                        f"SELECT Seq, Command FROM parjob WHERE Exitval is NULL LIMIT 1;"
                    )
                    row = cur.fetchone()
                    if not row:
                        log(f"{slot[workerid]}: No more job")
                        nomorejob = True
                        break

                    (
                        taskid,
                        cmd,
                    ) = row
                    cur.execute(
                        f"UPDATE parjob SET Starttime = unixepoch('now'), Exitval = -1000 WHERE Seq = {taskid};"
                    )
                    log(f"{slot[workerid]}: taskid, cmd:", taskid, cmd)
                    assert cur.rowcount == 1
                    con.commit()
                    break
                except Exception as e:
                    log(f"{slot[workerid]}: Exception:", e)
                    pass

        if nomorejob:
            break

        bashcmd = "bash -c '%s'" % cmd
        if verbose:
            print("%d: cmd:" % taskid, bashcmd)

        if not dryrun:
            starttime = time.time()
            with active.get_lock():
                active.value += 1

            p = subprocess.Popen(
                bashcmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True,
                env=os.environ.copy(),
            )
            with active.get_lock():
                active_ps[workerid] = p

            for line in iter(p.stdout.readline, ""):
                mq.put((workerid, taskid, line))

            p.wait()
            ## check out
            mq.put((workerid, taskid, None))
            with active.get_lock():
                active.value -= 1
                del active_ps[workerid]

            runtime = time.time() - starttime
            if verbose:
                print("%d: Done:" % taskid, p.returncode)

            with sqlite3.connect(dbfile) as con:
                cur = con.cursor()
                cur.execute(
                    f"UPDATE parjob SET Exitval = {p.returncode}, JobRuntime = {runtime} WHERE Seq = {taskid};"
                )
                con.commit()

    return 0


def jobcount():
    with sqlite3.connect(dbfile) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT count(1), sum(case when Exitval == 0 then 1 else 0 end) FROM parjob;"
        )
        row = cur.fetchone()
        (
            total,
            done,
        ) = row
    return (total, done)


def cmdlist(argv):
    """
    return list of list
    """
    cmds = list()
    _args = list()
    _type = 0  ## 0: regular, 1: file
    for x in argv:
        if x == ":::":
            cmds.append(_args)
            _args = list()
            _type = 0
        elif x == "::::":
            cmds.append(_args)
            _args = list()
            _type = 1
        else:
            if _type == 1:
                with open(x, "r") as f:
                    for line in f.readlines():
                        _args.append(line.rstrip())
            else:
                _args.append(x)

    cmds.append(_args)
    _args = list()

    return cmds


def progress(done, total, latest_line=False, progress=False):
    def putline():
        os.system("tput ll")
        print("\r", end="", flush=True)
        print(
            "Processing/Done/Total/Completed(%%)/Time(sec): %d/%d/%d/%.01f%%/%.02fs"
            % (
                active.value,
                done,
                total,
                float(done) / total * 100,
                time.time() - t0,
            ),
            end="",
            flush=True,
        )
        if not latest_line:
            print("")
        os.system("tput el")

    extra = 1 if progress else 0
    t0 = time.time()
    t1 = t0
    while True:
        workerid, taskid, line = mq.get()
        if (workerid is None) or (done == total):
            total, done = jobcount()
            putline()
            break

        if line is not None:
            if latest_line:
                os.system("tput ll")
                print("\r", end="", flush=True)
                os.system("tput sc")
                for i in range(slot[workerid] + extra):
                    os.system("tput cuu1")
                print("%d:" % taskid, line.rstrip(), end="", flush=True)
                os.system("tput el")
                os.system("tput rc")
            else:
                print("%d:" % taskid, line, end="", flush=True)

            if progress:
                ## try not too frequent
                if time.time() - t1 > 1:
                    total, done = jobcount()
                    putline()
                    t1 = time.time()
                else:
                    pass


def checkdb(args):
    with sqlite3.connect(dbfile) as con:
        # con.row_factory = sqlite3.Row
        cur = con.cursor()
        # cur.execute("SELECT count(1) as Total, sum(case when Exitval == 0 then 1 else 0 end) as Finished FROM parjob")
        if args.list:
            cur.execute(
                "SELECT Seq, "
                "datetime(Starttime, 'unixepoch', 'localtime') as Starttime, "
                "JobRuntime, Exitval, Command "
                "FROM parjob"
            )
            rows = cur.fetchall()
            format = " {:>4} {:<19} {:>9} {:>7} {:<80}"
            colnames = [desc[0] for desc in cur.description]
            bars = ["-" * len(desc[0]) for desc in cur.description]
            print(format.format(*colnames))
            print(format.format(*bars))
            for row in rows:
                print(
                    format.format(
                        *map(
                            lambda x: str(x)
                            if not isinstance(x, float)
                            else "%.2f" % x,
                            row,
                        )
                    )
                )
        else:
            cur.execute(
                "SELECT count(1) as Total, "
                "sum(case when Exitval == -1000 then 1 else 0 end) as Processing, "
                "sum(case when Exitval == 0 then 1 else 0 end) as Finished "
                "FROM parjob"
            )
            row = cur.fetchone()
            format = " {:>5} {:>10} {:>8}"
            colnames = [desc[0] for desc in cur.description]
            bars = ["-" * len(desc[0]) for desc in cur.description]
            print(format.format(*colnames))
            print(format.format(*bars))
            print(format.format(*map(str, row)))


def resetdb(args):
    with sqlite3.connect(dbfile) as con:
        cur = con.cursor()
        cur.execute("SELECT count(*) FROM parjob WHERE Exitval <> 0;")
        (count,) = cur.fetchone()
        ans = input("%d number of rows will be reset. Continue? (Y/N): " % count)
        if ans == "Y" or ans == "y":
            cur.execute("UPDATE parjob SET Exitval = NULL WHERE Exitval <> 0;")
            print("Rset:", cur.rowcount)
            con.commit()
        else:
            print("Aborted.")


def initdb(args):
    cmds = cmdlist(sys.argv[1:])
    args_list = cmds[1:]
    cmd = " ".join(args.cmd)
    ## check if cmd has valid formatter
    valid = any(a is not None or b is not None for _, a, b, _ in Formatter().parse(cmd))
    if not valid:
        cmd += " {}" * len(args_list)

    task_list = list()
    for i, argpair in enumerate(itertools.product(*args_list)):
        fullcmd = cmd.format(*argpair)
        task_list.append((i, fullcmd))

    dbfile = args.dbfile

    with sqlite3.connect(dbfile) as con:
        cur = con.cursor()
        try:
            sql = "DROP TABLE parjob;"
            cur.execute(sql)
        except:
            pass

        sql = (
            "CREATE TABLE parjob "
            "(Seq BIGINT,"
            " Host TEXT,"
            " Starttime FLOAT(44),"
            " JobRuntime FLOAT(44),"
            " Send BIGINT,"
            " Receive BIGINT,"
            " Exitval BIGINT,"
            " _Signal BIGINT,"
            " Command TEXT,"
            " V1 TEXT,"
            " Stdout TEXT,"
            " Stderr TEXT);"
        )
        cur.execute(sql)

        for i, cmd in task_list:
            sql = "INSERT INTO parjob (Seq,Command) VALUES (%d, '%s');" % (
                i,
                cmd,
            )
            cur.execute(sql)
        con.commit()
        res = cur.execute("select count(*) from parjob;")
        (ntotal,) = res.fetchone()
        print("%s created" % (dbfile))
        print("%d tasks added." % (ntotal))


def main(args):
    total, done = jobcount()
    p = threading.Thread(
        target=progress,
        args=(
            done,
            total,
            args.latest_line,
            args.progress,
        ),
    )
    p.start()

    env = os.environ.copy()
    counter = mp.Value("i", 0)
    # pool = ProcessPoolExecutor(max_workers=args.nworkers, initializer=hello, initargs=(counter,))
    pool = ThreadPoolExecutor(max_workers=args.nworkers, initializer=hello, initargs=(counter,))

    with pool as executor:
        future_list = list()
        for index in range(args.nworkers):
            future = executor.submit(execute, verbose=args.verbose, dryrun=args.dryrun)
            future_list.append(future)

        for future in future_list:
            future.result()

        mq.put((None, None, None))
        p.join()


if __name__ == "__main__":

    def usage():
        # print(
        #     "USAGE: %s <OPTIONS> [ ::: <ARGUMENTS> ]* [ :::: ARGFILE ]*" % (sys.argv[0])
        # )
        parser_main.print_help()
        # parser_args.print_help()
        sys.exit()

    parser_main = argparse.ArgumentParser(prog="OPTIONS", add_help=False)
    parser_main.add_argument("--dbfile", help="dbfile", default="pardb.sqlite")
    parser_main.add_argument("-v", "--verbose", action="store_true", help="verbose")

    subparsers = parser_main.add_subparsers(
        title="subcommands", description="valid subcommands", dest="command"
    )

    ## subcommand: check
    parser = subparsers.add_parser("check")
    parser.add_argument("-l", "--list", action="store_true", help="list")
    parser.set_defaults(func=checkdb)

    ## subcommand: reset
    parser = subparsers.add_parser("reset")
    parser.set_defaults(func=resetdb)

    ## subcommand: init
    parser = subparsers.add_parser("init")
    parser.set_defaults(func=initdb)
    parser.add_argument("cmd", help="command to execute", nargs=argparse.REMAINDER)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    ## subcommand: exec
    parser = subparsers.add_parser("exec")
    parser.set_defaults(func=main)
    parser.add_argument(
        "-j", "--nworkers", type=int, help="Number of workers", default=4
    )
    parser.add_argument("--progress", action="store_true", help="print progress")
    parser.add_argument(
        "--latest-line", action="store_true", help="print only last line"
    )
    parser.add_argument("--dryrun", action="store_true", help="dryrun")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    parser_args = argparse.ArgumentParser(prog="ARGUMENTS", add_help=False)
    parser_args.add_argument("args", help="arguments", nargs=argparse.REMAINDER)

    cmds = cmdlist(sys.argv[1:])
    args, _unknown = parser_main.parse_known_args(cmds[0])
    if len(_unknown) > 0:
        print("Unknown options:", _unknown)
        usage()

    if args.command == "init":
        args_cmd_list = list()
        for cmd in cmds[1:]:
            args_cmd, _unknown = parser_args.parse_known_args(cmd)
            if len(_unknown) > 0:
                usage()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    log("Python version:", ".".join(map(str, sys.version_info[:3])))
    log("Python info:", sys.version)
    args.func(args)
    sys.exit(0)
