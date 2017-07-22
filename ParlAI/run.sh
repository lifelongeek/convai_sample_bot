#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 
bs=32
OPTIND=1

train=1

while getopts "e:g:t:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

script='python examples/drqa/train.py -t squad'
script=${script}' --log_file '$exp_dir'/'${exp}'.log --expnum '${exp}
script=${script}' -bs '${bs}


if [ $train -eq 0 ]; then
	script='python examples/drqa/eval.py -t squad'
	script=${script}' --pretrained_model '${exp_dir}'/'${exp}' --datatype valid --expnum '${exp}
fi

case "$exp" in
    1) $script --model_file $exp_dir/$exp --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
		;;
    2) $script --model_file $exp_dir/$exp --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 --ans_sent_predict True --task_QA False
		;;
    3) $script --model_file $exp_dir/$exp --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --task_QA True
		;;
    4) $script --model_file $exp_dir/$exp --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --task_QA True --hidden_size_sent 256
        ;;
	5) $script --model_file $exp_dir/$exp --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.1
		;;


    9) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(1,5), (2,10), (3,25), (4,30), (5,55), (6,70)]'
		;;

    10) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(1,15), (2,30), (3,75), (4,90), (5,165), (6,210)]'
		;;
	
    11) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 0
		;;

    12) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 1

		;;
    13) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 2
		
		;;
    14) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 2
		;;

    15) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 0
		;;

    16) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
		;;

    17) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 0
                ;;

    18) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 1
		;;

    19) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 1
                ;;

    20) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,100)]' --nLayer_Highway 2
                ;;

    21) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;

    22) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 2
		;;

    23) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.5 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
		;;

    24) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
		;;
   25) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.1 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;
   26) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(6,200)]' --nLayer_Highway 1
                ;;
   27) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(5,300)]' --nLayer_Highway 1
                ;;
   28) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 1000 --gpu $gpuid --add_char2word True --kernels '[(4,300)]' --nLayer_Highway 1
                ;;
   29) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,250)]' --nLayer_Highway 1
                ;;
   30) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,300)]' --nLayer_Highway 1
                 ;;
   31) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,10), (4,20),(5,25),(6,35),(7,40)]' --nLayer_Highway 1
                ;;
   32) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,20), (4,35),(5,50),(6,65),(7,80)]' --nLayer_Highway 1
                ;;
   33) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,25), (4,40),(5,55),(6,70),(7,85)]' --nLayer_Highway 1
                ;;
   34) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,30), (4,45),(5,60),(6,75),(7,90)]' --nLayer_Highway 1
                ;;
   35) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 --vocab_size 30000
                ;;
   36) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,15), (4,30),(5,45),(6,60),(7,75)]' --nLayer_Highway 1 --vocab_size 30000
                ;;
   37) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 --vocab_size 50000
                ;;
  38) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,15), (4,30),(5,45),(6,60),(7,75)]' --nLayer_Highway 1 --vocab_size 50000
                ;;
  39) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 --vocab_size 70000
                ;;
  40) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(3,15), (4,30),(5,45),(6,60),(7,75)]' --nLayer_Highway 1 --vocab_size 70000
                ;;
  
   41) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.1 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;
   42) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.5 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;
   
   43) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;
   44) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1
                ;;

   # Sentence prediction start
  
   45) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --task_QA True --hidden_size_sent 128
                ;;
   46) $script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --hidden_size_sent 256
                ;;

   47)$script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.1
                ;;
   48)$script --model_file $exp_dir/$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --gpu $gpuid --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.01
				;;


esac
