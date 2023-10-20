CUDA_VISIBLE_DEVICES=4 torchrun --master_port 29905 --nproc_per_node 1 01-llama_questype.py --ckpt_dir ckpts/llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 1024 --max_batch_size 4
cd yolov7
CUDA_VISIBLE_DEVICES=4 python 02-detect-quesobj.py --weights ../ckpts/yolo-best.pt --conf 0.85 --img-size 640 --save-txt
cd ..
python 03-create-test.py
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --master_port=29909 --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2.yaml