python run_infer_wsi.py \
--gpu="0" \
--batch_size=25 \
--model="/root/romesco_workspace/resnet34_cerberus" \
--input_dir="wsi_test/" \
--output_dir="output_test/" \
--cache_path="/root/dgx_workspace/cache" \
--save_thumb