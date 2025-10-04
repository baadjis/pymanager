#!/bin/bash

function choose_from_menu() {
    local prompt="$1" outvar="$2"
    shift
    shift
    local options=("$@") cur=0 count=${#options[@]} index=0
    local esc=$(echo -en "\e") # cache ESC as test doesn't allow esc codes
    printf "$prompt\n"
    while true
      do
        # list all options (option list is zero-based)
        index=0
        for o in "${options[@]}"
        do
            if [ "$index" == "$cur" ]
            then echo -e " >\e[7m$o\e[0m" # mark & highlight the current option
            else echo "  $o"
            fi
            index=$(( $index + 1 ))
        done
        read -s -n3 key # wait for user to key in arrows or ENTER
        if [[ $key == ${esc}[A ]] # up arrow
        then cur=$(( $cur - 1 ))
            [ "$cur" -lt 0 ] && cur=0
        elif [[ $key == ${esc}[B ]] # down arrow
        then cur=$(( $cur + 1 ))
            [ "$cur" -ge "$count" ] && cur=$(( $count - 1 ))
        elif [[ $key == "" ]] # nothing, i.e the read delimiter - ENTER
        then 
           break
        fi
        echo -en "\e[${count}A" 
      done                                                                                                                  
    # export the selection to the requested output variable
    printf -v $outvar "${options[$cur]}"
}

function start_mongodb(){

mongod_status=$(sudo systemctl status mongod)

#echo "${mongod_status}"

if [[ "${mongod_status}" == *"active (running)"* ]]

then

echo "MongoDB Service is already running."

else

sudo systemctl start mongod

echo "Start MongoDB Service"

fi

}
function print_figlet(){
printf "%50s\n" |tr " " "="

figlet -t -k -f slant pymanager

printf "%50s\n" |tr " " "="
}
function run_cli(){
  local cliname=$1
  PWD=$(pwd)

function activate(){
    . $PWD/venv/bin/activate
}

activate
#export PYTHONPATH=$PYTHONPATH:$PWD
python cli/"$cliname".py
exit
}
clinames=("portfolio" 
"stock"
"index")
function launch_shell(){

print_figlet
#get_clinames 

choose_from_menu "choose shell" cliname "${clinames[@]}"

run_cli "$cliname"

}
start_mongodb
export PYTHONPATH=$PYTHONPATH:$PWD
launch_shell

