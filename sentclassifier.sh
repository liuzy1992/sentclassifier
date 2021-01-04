#!/usr/bin/env bash

model_name="svm"
num_jobs=1
outdir="."

name=$(realpath "$BASH_SOURCE")
path=$(dirname "$name")

declare -g -A __sourced__files__
if [[ ! -v __sourced__files__[$BASH_SOURCE] || $__force__source__ ]]; then
    __sourced__files__[$BASH_SOURCE]=$(realpath "$BASH_SOURCE")
    function sentclassifier {
        while [[ $# -gt 0 ]]; do
            case $1 in
                -h|--help)
                    cat "${name%.sh}_help.txt"
                    return
                    ;;
                -i)
                    infile="$2"
                    shift
                    shift
                    ;;
                -m)
                    model_name="$2"
                    shift
                    shift
                    ;;
                -o)
                    outdir="$2"
                    shift
                    shift
                    ;;
                -n)
                    num_jobs="$2"
                    shift
                    shift
                    ;;
                *)
                    echo "ERROR: Bad option '$1'." >&2
                    cat "${name%.sh}_help.txt"
                    return -1
                    ;;
            esac
        done
        
        if [ ! -d $outdir ]; then
            mkdir -p $outdir
        fi

        "$path"/run_pipeline.py $infile $model_name $outdir $num_jobs
    
    }

    if ! { ( return ) } 2>/dev/null; then
        set -e
        sentclassifier "$@" || exit
    fi

fi
