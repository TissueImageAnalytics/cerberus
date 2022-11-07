python run_infer_tile.py \
--gpu="0" \
--batch_size=25 \ # dependent on hardware
--model="/root/romesco_workspace/resnet34_cerberus" \
--input_dir="images_test/" \
--output_dir="output_test/"