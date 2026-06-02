#!/bin/bash

set -e
umask 002

BASE=/pscratch/sd/m/mlupopa/VASP_Calculations_Binaries_NbTaVHfTiZr/bcc_binaries_NbTaVHfTiZr/bcc_Nb-Ta-Zr

WDIR=$1
echo $WDIR
[ -z $WDIR ] && echo "Need WDIR" && exit -1

cd $WDIR
echo "Executing in $(pwd)"


function run_vasp {
    if [ -L POTCAR ] && [ ! -e POTCAR ]; then
        echo "POTCAR symlink is broken; recreating."
        rm -f POTCAR
    fi
    if [ ! -f POTCAR ]; then
        ln -s $BASE/POTCAR POTCAR || {
            echo "Failed to set up POTCAR link to $BASE/POTCAR"
            return 1
        }
    fi
    sleep $((1 + $RANDOM % 10))
    date
    time srun -N1 -n4 -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -u --cpu-bind=cores vasp_std 2>&1 | tee $1.out

    EVAL=${PIPESTATUS[0]}
    if grep -q "FIO-F-217/unformatted read/unit=12/attempt to read past end of file." $1.out; then
        echo "Detected unreadable WAVECAR. Removing it and retrying once."
        rm -f WAVECAR
        time srun -N1 -n4 -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -u --cpu-bind=cores vasp_std 2>&1 | tee -a $1.out
        EVAL=${PIPESTATUS[0]}
    fi
    if grep -q ieee_underflow $1.out; then
        # Warning: ieee_underflow is signaling
        echo "Found ieee_underflow. Ignore."
        EVAL=0
    fi
    return $EVAL
}

function has_null() {
	! ( tr -d '\0' <"$1" | diff -q - "$1" >/dev/null )
}

prefixes=(0 "" "N")
precision=(0 Low Normal)

function check_complete {
    n="$1"
    pre="${prefixes[n]}"

	has_out=0
	if [ -s CONTCAR ] || [ -s ${pre}0.CONTCAR ]; then
		has_out=1
	fi

    # No output yet. We are on step 0.
    if [ $has_out -eq 0 ]; then
        echo 0
        return 1
    fi

    # Find last-numbered $i.OUTCAR existing in this dir.
    for((i=9;i>=0;i--)); do
        [ -s $pre$i.CONTCAR ] && break
    done
	if [ $i -eq -1 ] || ( [ -s CONTCAR ] && ! diff -q CONTCAR $pre$i.CONTCAR >&2 ); then
        # nobody renamed the files yet
        let i=$i+1
        rename $pre$i >&2
    fi

    if grep -q "reached required accuracy - stopping structural energy minimisation" $pre$i.OUTCAR; then
        if [ ! -f "rlx$n.$i.out" ]; then
           # No log yet → don’t short-circuit; force one run to produce rlx<n>.<i>.out
           let i=i+1 # next output step
	   echo $i
           return 1 # no - not complete
        fi
        echo >&2 "Step $n complete (at $i)"
        cp rlx$n.$i.out rlx$n.out
        echo $i
        return 0 # yes - this is complete
    fi

    let i=i+1 # next output step
    echo $i
    return 1 # no - not complete
}

function rename {
	if ! [ -s CONTCAR ]; then
		echo >&2 "Missing/empty CONTCAR - aborting."
		exit 1
	fi
	mv INCAR $1.INCAR
	mv POSCAR $1.POSCAR
	mv OUTCAR $1.OUTCAR
	mv XDATCAR $1.XDATCAR
	cp CONTCAR $1.CONTCAR
	mv CONTCAR POSCAR
}

function step {
    n="$1"
    pre="${prefixes[n]}"

    i=`check_complete $n` && return
    # implement new naming convention
    [ -f rlx$n.out ] && ! [ -f rlx$n.0.out ] \
		&& mv rlx$n.out rlx$n.0.out

    sed "s/PRECISION/${precision[n]}/g" $BASE/START >INCAR
    run_vasp rlx$n.$i
	if has_null CONTCAR; then
		echo "VASP produced a bad CONTCAR!"
		rm CONTCAR
		exit 1
	fi

    rename $pre$i

    check_complete $n || {
        echo "Step $n incomplete!"
        exit 1
    }
}

if [ -f INCAR ] && grep -q "PREC *= *Normal" INCAR; then
    step 2
else
    step 1
    step 2
fi
