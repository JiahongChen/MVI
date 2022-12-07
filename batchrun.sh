python pretrained_model_rename.py
rm -r ./utils/__pycache__
rm -r ./model/__pycache__
python main.py --task CP --source Clipart --target Product 
sleep 60
