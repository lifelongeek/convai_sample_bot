#!/bin/bash

exp_dir='exp-squad'
emb='data/glove.840B.300d.txt'
exp=
gpuid= 

train=1 # train=1, eval=0

OPTIND=1
while getopts "e:g:t:" opt; do
	case "$opt" in
		e) exp=$OPTARG ;;
		g) gpuid=$OPTARG ;;
		t) train=$OPTARG ;;
	esac
done
shift $((OPTIND -1))

script='examples/drqa/train.py'
script=${script}' --log_file '$exp_dir'/exp'${exp}'.log'
script=${script}' -bs 32'
if [ $train -eq 1 ]; then
	script=${script}' --train_interval 3368'
fi

if [ $train -eq 0 ]; then
	script='examples/drqa/eval.py'
	script=${script}' --pretrained_model '${exp_dir}/exp${exp}' --datatype valid'
fi

case "$exp" in
	0) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --gpu $gpuid --tune_partial 1000 
		;;
	2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid
		;;
	3) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000
		;;
	4) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False
		;;
	5) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --rnn_padding True
		;;
	6) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False --rnn_padding True
		;;
	7) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --fix_embeddings False --rnn_padding True --optimizer adam --learning_rate 0.005
		;;
	8) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --rnn_padding True
		;;
	h11) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 ## r-net, _dep
		;;
	h11-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --hidden_size 75 ## r-net, _dep
		;;
	h12) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --hidden_size 75 ## r-net, GatedAttentionRNN
		;;
	h12-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --hidden_size 75 --rnn_padding True ## r-net, GatedAttentionRNN
		;;
	h12-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 -bs 16  ## r-net, GatedAttentionRNN
		;;
	h13) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --qp_bottleneck True ## r-net, GatedAttentionRNN
		;;
	h13-1) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000 --qp_bottleneck True --use_qemb False ## r-net, GatedAttentionRNN
		;;
	h13-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000  ## r-net, GatedAttentionRNN
		;;
	h13-3) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000  --use_qemb False ## r-net, GatedAttentionRNN
		;;
	h13-fix) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True ## r-net, GatedAttentionRNN
		;;
	h13-fix-bi) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True  ## r-net, GatedAttentionRNN
		;;
###########################
		## h13-fix-bi-ldecay
###########################
	h13-fix-bi-ldecay) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
	;;
	h13-fix-bi-ldecay-h75) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --hidden_size 75
		;;
	h13-fix-bi-ldecay-4) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.4 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-5) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.5 --dropout_emb 0.5 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.2 --dropout_emb 0.2 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-2-0) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.2 --dropout_emb 0 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-3-0) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-4-0) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True 
		;;
	h13-fix-bi-ldecay-s3435) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --random_seed 3435
		;;
	h13-fix-bi-ldecay-s8031) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --random_seed 8031 ## OOM
		;;
	h13-fix-bi-ldecay-pad) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --rnn_padding True
		;;
	h13-fix-bi-ldecay-gru) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --rnn_type gru ## OOM
		;;
	h13-fix-bi-ldecay-gru2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.2 --dropout_emb 0 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --rnn_type gru  ## OOM
		;;
	h13-fix-bi-ldecay-adam) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --learning_rate 0.0005 --optimizer adam # --train_interval 3368
		;;
	h13-fix-bi-ldecay-adam-clipx) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --learning_rate 0.0005 --optimizer adam --grad_clipping 0
		;;
	h13-fix-bi-ldecay-clipx) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --grad_clipping 0
		;;
	h13-fix-concat) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_concat True
		;;
	h13-fix-bi-concat) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True
		;;
	h13-fix-bi-concat-ldecay) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True --lrate_decay True
		;;
		######
		######
		###### PP matching
	h14-fix) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24 --rnn_type gru ## QP -> UNI PP -> UNI
		;;
	h14-fix-2) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24
		;;
	h14-fix-lr) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24 --rnn_type gru
		;;
	h14-fix-2-lr) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --pp_bottleneck True --hidden_size 75 -bs 24
		;;
		#### AFTER PP update by GM
	h14-bt-gt-rt) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True
		;;
	h14-bt-gt-rx) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_rnn False
		;;
	
	h14-bt-gt-rt-if) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False
		;;
	h14-bt256-gt-rt-if) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256
		;;
	h14-bt256-gt-rt-if-qpcat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256 --qp_concat True
		;;
	h14-bt256-gt-rt-if-ppcat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256 --pp_concat True
		;;
	h14-bt256-gt-rt-if-qpcat-ppcat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256 --qp_concat True --pp_concat True
		;;
	h14-bt128-gt-rt-if-qpcat-ppcat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 128 --qp_concat True --pp_concat True
		;;
	h14-bt256-gt-rt-if-qpcat-ppcat-bs64) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256 --qp_concat True --pp_concat True -bs 64
		;;
	h14-bt128-gt-rt-if-qpcat-ppcat-bs64) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 128 --qp_concat True --pp_concat True -bs 64
		;;
		#	h14-fix-concat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_concat True --pp_concat True
#		;;
#	h14-fix-bi-concat) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --qp_concat True --pp_birnn True --pp_concat True
#		;;
		######
		######
		###### Char + QP matching
	h15-fix-bi-ldecay-44) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --tune_partial 0 --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 -bs 20 
	;;
	h15-fix-bi-ldecay-44-2) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --tune_partial 0 --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 -bs 20 
	;;

	h15-bt256-gt-rt-if-qpcat-ppcat-chm) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256 --qp_concat True --pp_concat True --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 
		;;
	h15-bt256-gt-rt-if-qpcat-ppcat-chl) python $script -t squad --net rnet --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --pp_bottleneck True --pp_gate True --pp_identity False --hidden_size_bottleneck 256--qp_concat True --pp_concat True --add_char2word True --kernels '[(1, 15), (2, 20), (3, 35), (4, 40), (5, 75), (6, 90)]' --nLayer_Highway 1
		;;


	h15-fix-bi-ldecay-sent-char) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.4 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --tune_partial 0 --add_char2word True --kernels '[(5,200)]' --nLayer_Highway 1 --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.01
	;;
	h15-fix-bi-ldecay-49) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.1  
	;;
	h15-fix-bi-ldecay-50) python $script -t squad --model_file $exp_dir/exp$exp --embedding_file $emb --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --qp_bottleneck True --qp_birnn True --lrate_decay True --dropout_rnn 0.4 --dropout_emb 0.3 --tune_partial 0 --ans_sent_predict True --hidden_size_sent 256 --coeff_ans_predict 0.01
	;;



	debug) python $script -t squad --model_file $exp_dir/exp$exp --dropout_rnn 0.3 --dropout_emb 0.3 --gpu $gpuid --tune_partial 1000   ## r-net For debug
esac

