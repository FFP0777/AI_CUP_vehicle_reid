訓練reid:
cd vehicle_reid
pip install -r requirements.txt
在終端機下輸入:python train.py --data_dir ../datasets  --name resnet50_ibn11 --color_jitter  --mixstyle  --train_csv_path ../datasets/AI_CUP/train_labels.csv  --val_csv_path ../datasets/AI_CUP/val_labels.csv  --batchsize 32 --total_epoch 120 --warm_epoch 3  --save_freq 5 --erasing_p 0.5 --model resnet_ibn --model_subtype 50  --contrast   --num_workers 0 --samples_per_class=4 --cosine --checkpoint C:\Users\LPCAS\Desktop\vehicle_reid\vehicle_reid\model\resnet50_mixstyle\net_19.pth --start_epoch 20 --lr=0.008

使用的預訓練模型(來源:https://github.com/regob/vehicle_mtmc?tab=readme-ov-file)
https://drive.google.com/file/d/1STbsacssLtlHpUesNzuTeUPrfMlWbSKu/view
我就是用這個來訓練我的AI_CUP模型

如果想使用訓練完的模型看query可視化效果及查看rank,map，我已分割好test_csv及query_csv:
打開.ipynb_checkpoints/run_ok.ipynb
照順序執行即可，如果要用不同ID車輛進行query，[19]在random.seed(666)中的666替換任意數字就可了 。

