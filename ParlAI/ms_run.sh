#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0
debug=0

OPTIND=1
while getopts "e:g:t:d:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
		d) debug=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

script='examples/drqa/train_MSmarco.py'
script=${script}' --log_file '$exp_dir'/exp'${exp}'.log'
script=${script}' -bs 32 -t squad'
if [ $train -eq 1 ]; then
	script=${script}' --train_interval 3368'
fi

if [ $train -eq 0 ]; then
	script='examples/drqa/eval.py -t squad'
	script=${script}' --pretrained_model '${exp_dir}/${exp}' --datatype valid'
fi

#if [ $debug -eq 0 ]; then
#	script=${script}' --embedding_file '$emb
#fi


# lrate_decay=True by default
case "$exp" in
	#echo $exp_dir/$exp
	#echo $emb
	#echo $gpuid

	DEBUG_msmarco) python $script --model_file $exp_dir/$exp --gpu $gpuid --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --hidden_size 16 --hidden_size_bottleneck 16 --msmarco_paragraph_concat True
	;;

esac

