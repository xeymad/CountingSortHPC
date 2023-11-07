#!/bin/bash
# 
# Course: High Performance Computing 2020/2021
# 
# Lecturer: Francesco Moscato	fmoscato@unisa.it
#
# Group:
# Capitani	Giuseppe	0622701085	g.capitani@studenti.unisa.it               
# Falanga	Armando	0622701140  a.falanga13@studenti.unisa.it 
# Terrone	Luigi		0622701071  l.terrone2@studenti.unisa.it
#
# Copyright (C) 2021 - All Rights Reserved 
#
# This file is part of CommonAssignmentMPI01.
#
# CommonAssignmentMPI01 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CommonAssignmentMPI01 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CommonAssignmentMPI01.  If not, see <http://www.gnu.org/licenses/>.
#

TIME_STAMP=$(date +%s)
NMEASURES=50

ARRAY_RC=(5000000 10000000 20000000)
ARRAY_THS=(0 1 2 4 8)
ARRAY_VERSION=(1 2 3 4)

trap "exit" INT

# Get script path
SCRIPTPATH=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )

for size in "${ARRAY_RC[@]}"; do
    $1/init_vect $size
	for ver in "${ARRAY_VERSION[@]}"; do
		for ths in "${ARRAY_THS[@]}"; do
			ths_str=$(printf "%02d" $ths)
			
			OUT_FILE=$SCRIPTPATH/measure/SIZE-$size-V$ver/SIZE-$size-NP-$ths_str-V$ver.csv

			mkdir -p $(dirname $OUT_FILE) 2> /dev/null
			
			echo $(basename $OUT_FILE)
			if [[ $ver -gt 1 && $ths -eq 0 ]]; then
				OLD_OUT_FILE=$SCRIPTPATH/measure/SIZE-$size-V1/SIZE-$size-NP-$ths_str-V1.csv
				ln -srf -T $OLD_OUT_FILE $OUT_FILE
				echo Created symbolic link to $(basename $OLD_OUT_FILE)
				continue
			fi
			echo "array_size,processes,read_time,global_counting_time,global_elapsed" >$OUT_FILE
			
			for ((i = 0 ; i < $NMEASURES	; i++)); do
				if [[ $ths -eq 0 ]]; then
					$1/program_seq $size  >> $OUT_FILE
					printf "\r> %d/%d %3.1d%% " $(expr $i + 1) $NMEASURES $(expr \( \( $i + 1 \) \* 100 \) / $NMEASURES)
					printf "#%.0s" $(seq -s " " 1 $(expr \( $i \* 40 \) / $NMEASURES))
				else
					mpirun.mpich -np $ths $1/program_parallel_V$ver $size >> $OUT_FILE
					printf "\r> %d/%d %3.1d%% " $(expr $i + 1) $NMEASURES $(expr \( \( $i + 1 \) \* 100 \) / $NMEASURES)
					printf "#%.0s" $(seq -s " " 1 $(expr \( $i \* 40 \) / $NMEASURES))
				fi
			done
			printf "\n"
		done
	done
done
